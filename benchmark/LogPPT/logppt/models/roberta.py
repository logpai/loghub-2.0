from transformers import (
    RobertaForMaskedLM,
    RobertaForTokenClassification,
    AutoTokenizer
)
from collections import Counter


def load_roberta(model_path, mode="prompt"):
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, add_prefix_space=True, do_lower_case=False)
    if mode == "prompt-tuning":
        model = RobertaForMaskedLM.from_pretrained(model_path)
        tokenizer.model_max_length = model.config.max_position_embeddings - 2
        return tokenizer, model
    elif mode == "fine-tuning":
        model = RobertaForTokenClassification.from_pretrained(model_path)
        tokenizer.model_max_length = model.config.max_position_embeddings - 2
        return tokenizer, model
    else:
        raise NotImplementedError("Choose from prompt-tuning or fine-tuning")


def add_label_token_roberta(model, tokenizer, label_map, wo_label_words=False):
    sorted_add_tokens = sorted(list(label_map.keys()), key=lambda x: len(x), reverse=True)
    # tokenizer.add_tokens(sorted_add_tokens)
    num_tokens, _ = model.roberta.embeddings.word_embeddings.weight.shape
    model.resize_token_embeddings(len(sorted_add_tokens) + num_tokens)
    if wo_label_words:
        return model
    for token in sorted_add_tokens:
        if token.startswith('i-'):
            index = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(token))
            if len(index) > 1:
                raise RuntimeError(f"{token} wrong split: {index}")
            else:
                index = index[0]
            # assert index >= num_tokens, (index, num_tokens, token)
            # n_label_word = len(list(set(label_map[token])))
            ws = label_map[token]
            print(tokenizer.convert_ids_to_tokens(ws))
            e_token = model.roberta.embeddings.word_embeddings.weight.data[ws[0]]
            for i in ws[1:]:
                e_token += model.roberta.embeddings.word_embeddings.weight.data[i]
            e_token /= len(ws)
            model.roberta.embeddings.word_embeddings.weight.data[index] = e_token

    return model
