from logppt.models.bert import load_bert, add_label_token_bert
from logppt.models.roberta import load_roberta, add_label_token_roberta
from logppt.models.xlnet import load_xlnet, add_label_token_xlnet
from logppt.models.gpt2 import load_gpt2, add_label_token_gpt2


def load_model(model_path, model_name, mode="prompt-tuning"):
    if model_name == "roberta":
        return load_roberta(model_path, mode)
    elif model_name == "bert":
        return load_bert(model_path, mode)
    elif model_name == "xlnet":
        return load_xlnet(model_path, mode)
    elif model_name == "gpt2":
        return load_gpt2(model_path, mode)
    else:
        raise NotImplementedError("Not implemented yet")


def add_label_token(lm_name, model, tokenizer, label_map, wo_label_words=False):
    if "roberta" in lm_name:
        return add_label_token_roberta(model, tokenizer, label_map, wo_label_words)
    elif "bert" in lm_name:
        return add_label_token_bert(model, tokenizer, label_map, wo_label_words)
    elif "gpt2" in lm_name:
        return add_label_token_gpt2(model, tokenizer, label_map, wo_label_words)
    elif "xlnet" in lm_name:
        return add_label_token_xlnet(model, tokenizer, label_map, wo_label_words)
    else:
        raise NotImplementedError("Not implemented yet")
