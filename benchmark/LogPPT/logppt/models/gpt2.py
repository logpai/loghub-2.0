from collections import Counter

from transformers import (
    GPT2LMHeadModel,
    GPT2ForTokenClassification,
    AutoTokenizer
)


def load_gpt2(model_path, mode="prompt"):
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, add_prefix_space=True, do_lower_case=False)
    tokenizer.pad_token = tokenizer.eos_token
    if mode == "prompt-tuning":
        model = GPT2LMHeadModel.from_pretrained(model_path)
        return tokenizer, model
    elif mode == "fine-tuning":
        model = GPT2ForTokenClassification.from_pretrained(model_path)
        return tokenizer, model
    else:
        raise NotImplementedError("Choose from prompt-tuning or fine-tuning")


def add_label_token_gpt2(model, tokenizer, label_map):
    sorted_add_tokens = sorted(list(label_map.keys()), key=lambda x: len(x), reverse=True)
    num_tokens, _ = model.transformer.wte.weight.shape
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
            ws = [x[0] for x in Counter(label_map[token]).most_common(n=5)]
            e_token = model.transformer.wte.weight.data[ws[0]]
            for i in ws[1:]:
                e_token += model.transformer.wte.weight.data[i]
            e_token /= len(ws)
            model.transformer.wte.weight.data[index] = e_token

    return model
