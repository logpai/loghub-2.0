import time

import torch


def compute_metrics(metric):
    results = metric.compute()
    final_results = {}
    for key, value in results.items():
        if isinstance(value, dict):
            for n, v in value.items():
                final_results[f"{key}_{n}"] = v
        else:
            final_results[key] = value
    return final_results


def get_labels(predictions, references, tokens, device, label_token_id_to_label, id_to_label, tokenizer,
               mode="prompt-tuning"):
    # Transform predictions and references tensos to numpy arrays
    if device.type == "cpu":
        y_pred = predictions.detach().clone().numpy()
        y_true = references.detach().clone().numpy()
        x_tokens = tokens.detach().clone().tolist()
    else:
        y_pred = predictions.detach().cpu().clone().numpy()
        y_true = references.detach().cpu().clone().numpy()
        x_tokens = tokens.detach().cpu().clone().tolist()

    # Remove ignored index (special tokens)
    # Here we only use the first token of each word for evaluation.
    if mode == "prompt-tuning":
        true_predictions = [
            [label_token_id_to_label[p].upper() if p in label_token_id_to_label.keys() else "O" for (p, l) in
             zip(pred, gold_label) if l != -100]
            for pred, gold_label in zip(y_pred, y_true)
        ]
    else:
        true_predictions = [
            [id_to_label[p].upper() for (p, l) in zip(pred, gold_label) if l != -100]
            for pred, gold_label in zip(y_pred, y_true)
        ]

    true_labels = [
        [id_to_label[l].upper() for (p, l) in zip(pred, gold_label) if l != -100]
        for pred, gold_label in zip(y_pred, y_true)
    ]

    ori_tokens = [
        [tokenizer.convert_ids_to_tokens(t) for (p, l, t) in zip(pred, gold_label, token) if l != -100]
        for pred, gold_label, token in zip(y_pred, y_true, x_tokens)
    ]

    true_labels = [['I-{}'.format(l[2:].upper()) if l != "O" else "O" for l in label] for label in true_labels]
    return true_predictions, true_labels, ori_tokens


def evaluate(metric, model, tokenizer, eval_dataloader, accelerator, pad_to_max_length, label_token_id_to_label,
             id_to_label, mode="prompt-tuning"):
    model.eval()
    device = accelerator.device
    start = time.time()
    token_list = []
    y_true = []
    y_pred = []
    for step, batch in enumerate(eval_dataloader):
        with torch.no_grad():
            ner_label = batch.pop('ori_labels', 'not found ner_labels')
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch, output_hidden_states=True)

        predictions = outputs.logits.argmax(dim=-1)

        labels = ner_label
        token_labels = batch.pop("input_ids")
        if not pad_to_max_length:  # necessary to pad predictions and labels for being gathered
            predictions = accelerator.pad_across_processes(predictions, dim=1, pad_index=-100)
            labels = accelerator.pad_across_processes(labels, dim=1, pad_index=-100)
            token_labels = accelerator.pad_across_processes(token_labels, dim=1, pad_index=-100)
        predictions_gathered = accelerator.gather(predictions)
        labels_gathered = accelerator.gather(labels)
        token_labels_gathered = accelerator.gather(token_labels)
        preds, refs, tokens = get_labels(predictions_gathered, labels_gathered, token_labels_gathered, device,
                                         label_token_id_to_label, id_to_label, tokenizer, mode)

        token_list.extend(tokens)
        y_true.extend(refs)
        y_pred.extend(preds)

        metric.add_batch(
            predictions=preds,
            references=refs,
        )  # predictions and preferences are expected to be a nested list of labels, not label_ids

    # eval_metric = metric.compute()
    eval_metric = compute_metrics(metric)

    print("Decoding time: {}s".format(time.time() - start))
    label = "overall"
    print("{}: {}, {}: {}, {}: {}, {}: {}".format(label + "_precision", eval_metric[label + "_precision"],
                                                  label + "_recall", eval_metric[label + "_recall"],
                                                  label + "_f1", eval_metric[label + "_f1"],
                                                  label + "_accuracy", eval_metric[label + "_accuracy"]))

    return eval_metric
