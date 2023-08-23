import logging
import math

import datasets
import torch
import transformers
from datasets import load_metric

from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    HfArgumentParser,
    AutoConfig,
    default_data_collator,
    get_scheduler,
    set_seed
)
from logppt.models import load_model
from accelerate import Accelerator
import copy

from logppt.utils import MainArguments, ModelArguments, TrainArguments, TaskArguments, find_labels
from logppt.data import load_data_parsing, load_data_anomaly_detection, CustomDataCollator
from logppt.models import add_label_token
from logppt.tokenization import parsing_tokenize_dataset
from logppt.evaluation import evaluate
from logppt.tasks.log_parsing import template_extraction

logger = logging.getLogger(__name__)
accelerator = Accelerator()

filter_list = ["and", "or", "the", "a", "of", "to", "at"]


def train():
    total_batch_size = train_args.per_device_train_batch_size * accelerator.num_processes * train_args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {train_args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {train_args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {train_args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {train_args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(train_args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    # model_path = "checkpoint_best.pt"
    # early_stopping = EarlyStopping(patience=5, verbose=False, path=model_path)
    for epoch in range(train_args.num_train_epochs):
        model.train()
        total_loss = []
        for step, batch in enumerate(train_dataloader):
            batch.pop('ori_labels', 'not found ner_labels')
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            total_loss.append(float(loss))
            loss = loss / train_args.gradient_accumulation_steps
            accelerator.backward(loss)
            if step % train_args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
                progress_bar.set_description(f"Loss: {float(loss)}")
                completed_steps += 1

            if completed_steps >= train_args.max_train_steps:
                break

        # early_stopping(np.average(total_loss), model)
        # if early_stopping.early_stop:
        #     print("Early stopping!!")
        #     break

    # load the last checkpoint with the best model
    # model.load_state_dict(torch.load(model_path))

    # Use the result of the last epoch
    best_metric = evaluate(metric, model, tokenizer, eval_dataloader, accelerator,
                           main_args.pad_to_max_length,
                           label_token_id_to_label, id_to_label, main_args.mode)
    print("Finish training, best metric: ")
    try:
        logger.info(best_metric)
    except Exception as _:
        print(best_metric)

    if main_args.output_dir is not None:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(main_args.output_dir, save_function=accelerator.save)
        tokenizer.save_pretrained(main_args.output_dir)

    if task_args.task_name == "log-parsing":
        template_extraction(tokenizer, model, accelerator, task_args.log_file, max_length=main_args.max_length,
                            model_name=model_type, shot=main_args.shot, dataset_name=task_args.dataset_name,
                            o_dir=task_args.task_output_dir, mode=main_args.mode)
    else:
        raise ValueError("Please choose the \"log-parsing\" task")


if __name__ == '__main__':
    parser = HfArgumentParser((MainArguments, ModelArguments, TrainArguments, TaskArguments))
    main_args, model_args, train_args, task_args = parser.parse_args_into_dataclasses()
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state)
    logger.setLevel(logging.INFO if accelerator.is_local_main_process else logging.ERROR)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if main_args.seed is not None:
        set_seed(main_args.seed)

    # Get the datasets: the data file are JSON files
    if task_args.task_name == "log-parsing":
        raw_datasets, text_column_name, label_name = load_data_parsing(main_args)
    else:
        raw_datasets, text_column_name, label_name = load_data_anomaly_detection(main_args)

    if model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path)
        model_type = config.model_type
        tokenizer, model = load_model(model_args.model_name_or_path, model_type, mode=main_args.mode)
    else:
        raise ValueError("missing model path")

    if task_args.task_name == "log-parsing":
        ori_label_token_map = {"i-val": []}
    else:
        ori_label_token_map = {"i-normal": [], "i-abnormal": []}
    sorted_add_tokens = sorted(list(ori_label_token_map.keys()), key=lambda x: len(x), reverse=True)
    tokenizer.add_tokens(sorted_add_tokens)

    label_list = list(ori_label_token_map.keys())
    label_list += 'o'

    label_to_id = {'o': 0}
    for label in label_list:
        if label != 'o':
            label_to_id[label] = len(label_to_id)
    num_labels = len(label_list)
    print("ori label:", ori_label_token_map)
    new_label_to_id = copy.deepcopy(label_to_id)
    label_to_id = new_label_to_id
    id_to_label = {id: label for label, id in label_to_id.items()}
    print("label to id:", label_to_id)
    print("id to label:", id_to_label)

    label_token_map = {item: item for item in ori_label_token_map}
    # label_token_map = ori_label_token_map
    print("label token map:", label_token_map)
    label_token_to_id = {label: tokenizer.convert_tokens_to_ids(label_token) for label, label_token in
                         label_token_map.items()}
    label_token_id_to_label = {idx: label for label, idx in label_token_to_id.items()}
    print("label token to id:", label_token_to_id)
    print("label token id to label:", label_token_id_to_label)
    # Preprocessing the datasets.
    # First we tokenize all the texts.
    padding = "max_length" if main_args.pad_to_max_length else False

    if task_args.task_name == "log-parsing":
        processed_raw_datasets, label_words, keywords = parsing_tokenize_dataset(tokenizer, raw_datasets,
                                                                                 text_column_name,
                                                                                 label_name, main_args.max_length,
                                                                                 padding,
                                                                                 label_to_id, label_token_to_id,
                                                                                 model_type,
                                                                                 main_args.mode)
        train_dataset = processed_raw_datasets["train"]
        eval_dataset = processed_raw_datasets["validation"]
    else:
        raise NotImplementedError()
    print(processed_raw_datasets)

    # DataLoaders creation:
    if main_args.pad_to_max_length:
        # If padding was already done ot max length, we use the default data collator that will just convert everything
        # to tensors.
        data_collator = default_data_collator
    else:
        # Otherwise, `CustomDataCollator` will apply dynamic padding for us (by padding to the
        # maximum length of the samples passed). When using mixed precision, we add `pad_to_multiple_of=8` to pad all
        # tensors to multiple of 8s, which will enable the use of Tensor Cores on NVIDIA hardware with compute
        # capability >= 7.5 (Volta).
        data_collator = CustomDataCollator(
            tokenizer, pad_to_multiple_of=None
        )

    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=data_collator, batch_size=train_args.per_device_train_batch_size
    )
    if eval_dataset is not None:
        eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator,
                                     batch_size=train_args.per_device_eval_batch_size)
    else:
        eval_dataloader = None
    if main_args.mode == "prompt-tuning":
        label_words = find_labels(model, train_dataloader, eval_dataloader, keywords)
        selected_words = []
        current_list = {i: 0 for i in range(100)}
        lbl_word2 = copy.deepcopy(label_words)
        lbl_word_indices = list(label_words.keys()).copy()
        for k in lbl_word_indices:
            token = tokenizer.convert_tokens_to_string([tokenizer.convert_ids_to_tokens(k)]).strip()
            if k in tokenizer.all_special_ids or len(token) < 3 or token in filter_list \
                    or token.count(token[0]) == len(token) or token in selected_words:
                del label_words[k]
            else:
                selected_words.append(token)
                current_list[len(token.strip())] += 1
        label_words = {'i-val': list(label_words.keys())[:train_args.no_label_words]}
        print(tokenizer.convert_ids_to_tokens(label_words['i-val']))
        model = add_label_token(model_type, model, tokenizer, label_words, train_args.wo_label_words)

    # dev_dataloader = DataLoader(dev_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)
    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": train_args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]

    optimizer = torch.optim.AdamW(
        optimizer_grouped_parameters,
        lr=train_args.learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8,
    )
    device = accelerator.device
    model.to(device)

    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader
    )

    label_id_list = torch.tensor([label_token_to_id[id_to_label[i]] for i in range(len(id_to_label)) if
                                  i != 0 and not id_to_label[i].startswith("B-")], dtype=torch.long, device=device)

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / train_args.gradient_accumulation_steps)
    if train_args.max_train_steps is None:
        train_args.max_train_steps = train_args.num_train_epochs * num_update_steps_per_epoch
    else:
        train_args.num_train_epochs = math.ceil(train_args.max_train_steps / num_update_steps_per_epoch)

    lr_scheduler = get_scheduler(
        name=train_args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=train_args.num_warmup_steps,
        num_training_steps=train_args.max_train_steps,
    )
    metric = load_metric("logppt/evaluation/seqeval_metric.py")
    train()
