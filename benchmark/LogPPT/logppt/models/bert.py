from collections import Counter

from transformers import (
    BertForMaskedLM,
    BertForTokenClassification,
    AutoTokenizer
)


def load_bert(model_path, mode="prompt"):
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, do_lower_case=False)
    if mode == "prompt-tuning":
        model = BertForMaskedLM.from_pretrained(model_path)
        return tokenizer, model
    elif mode == "fine-tuning":
        model = BertForTokenClassification.from_pretrained(model_path)
        return tokenizer, model
    else:
        raise NotImplementedError("Choose from prompt-tuning or fine-tuning")


def add_label_token_bert(model, tokenizer, label_map):
    sorted_add_tokens = sorted(list(label_map.keys()), key=lambda x: len(x), reverse=True)
    num_tokens, _ = model.bert.embeddings.word_embeddings.weight.shape
    model.resize_token_embeddings(len(sorted_add_tokens) + num_tokens)
    # return tokenizer
    for token in sorted_add_tokens:
        if token.startswith('i-'):
            index = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(token))
            if len(index) > 1:
                raise RuntimeError(f"{token} wrong split: {index}")
            else:
                index = index[0]
            # assert index >= num_tokens, (index, num_tokens, token)
            # n_label_word = len(list(set(label_map[token])))
            ws = [x[0] for x in Counter(label_map[token]).most_common(n=5)]
            e_token = model.bert.embeddings.word_embeddings.weight.data[ws[0]]
            for i in ws[1:]:
                e_token += model.bert.embeddings.word_embeddings.weight.data[i]
            e_token /= len(ws)
            model.bert.embeddings.word_embeddings.weight.data[index] = e_token

    return model
