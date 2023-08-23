import re


def get_parameter_list(s, template_regex):
    """
    :param s: log message
    :param template_regex: template regex with <*> indicates parameters
    :return: list of parameters
    """
    # template_regex = re.sub(r"<.{1,5}>", "<*>", template_regex)
    if "<*>" not in template_regex:
        return []
    template_regex = re.sub(r'([^A-Za-z0-9])', r'\\\1', template_regex)
    template_regex = re.sub(r'\\ +', r'\\s+', template_regex)
    template_regex = "^" + template_regex.replace("\<\*\>", "(.*?)") + "$"
    parameter_list = re.findall(template_regex, s)
    parameter_list = parameter_list[0] if parameter_list else ()
    parameter_list = list(parameter_list) if isinstance(parameter_list, tuple) else [parameter_list]
    return parameter_list


def parsing_tokenize_dataset(tokenizer, dataset, text_column_name, label_column_name, max_length, padding, label_to_id,
                             label_token_to_id, model_type="roberta", mode="prompt-tuning"):
    label_words = {
        "i-val": []
    }
    keywords = set([])

    def tokenize_and_align_labels(examples):
        examples[text_column_name] = [" ".join(x.strip().split()) for x in examples[text_column_name]]
        tokenized_inputs = tokenizer(
            examples[text_column_name],
            max_length=max_length,
            padding=padding,
            truncation=True,
            is_split_into_words=False,
        )
        target_tokens = []
        labels = []
        t_token = "i-val"
        for i, label in enumerate(examples[label_column_name]):
            content = examples[text_column_name][i]
            label = " ".join(label.strip().split())
            variable_list = get_parameter_list(content, label)
            input_ids = tokenized_inputs.input_ids[i]
            input_tokens = tokenizer.convert_ids_to_tokens(input_ids)

            label_ids = []
            target_token = []
            processing_variable = False
            variable_token = ""
            if model_type == 'bert':
                input_tokens = [tokenizer.convert_tokens_to_string([x]) for x in input_tokens]
                input_tokens = [" " + x if "##" not in x else x[2:] for x in input_tokens]
            elif model_type == 'xlnet':
                input_tokens = [" " + x[1:] if "▁" in x else x for x in input_tokens]
            else:
                input_tokens = [tokenizer.convert_tokens_to_string([x]) for x in input_tokens]
            # pos = 0
            for ii, (input_idx, input_token) in enumerate(zip(input_ids, input_tokens)):
                if input_idx in tokenizer.all_special_ids:
                    target_token.append(-100)
                    label_ids.append(-100)
                    continue
                # Set target token for the first token of each word.
                if (label[:3] == "<*>" or label[:len(input_token.strip())] != input_token.strip()) \
                        and processing_variable is False:
                    processing_variable = True
                    # print("===================")
                    # print(content)
                    # print(label)
                    # print(variable_list)
                    # print(input_tokens)
                    # print("===================")
                    variable_token = variable_list.pop(0).strip()
                    pos = label.find("<*>")
                    label = label[label.find("<*>") + 3:].strip()
                    input_token = input_token.strip()[pos:]

                if processing_variable:
                    input_token = input_token.strip()
                    if input_token == variable_token[:len(input_token)]:
                        if mode == "prompt-tuning":
                            target_token.append(label_token_to_id[t_token])
                        else:
                            target_token.append(1)
                        label_ids.append(label_to_id[t_token])
                        variable_token = variable_token[len(input_token):].strip()
                        # print(variable_token, "+++", input_token)
                    elif len(input_token) > len(variable_token):
                        if mode == "prompt-tuning":
                            target_token.append(label_token_to_id[t_token])
                        else:
                            target_token.append(1)
                        label_ids.append(label_to_id[t_token])
                        label = label[len(input_token) - len(variable_token):].strip()
                        variable_token = ""
                    else:
                        raise ValueError(f"error at {variable_token} ---- {input_token}")
                    if len(input_token) >= 3:
                        label_words[t_token].append(input_idx)
                    if len(variable_token) == 0:
                        processing_variable = False
                else:
                    keywords.add(input_idx)
                    input_token = input_token.strip()
                    if input_token == label[:len(input_token)]:
                        if mode == "prompt-tuning":
                            target_token.append(input_idx)
                        else:
                            target_token.append(0)
                        label_ids.append(label_to_id['o'])
                        label = label[len(input_token):].strip()
                    else:
                        raise ValueError(f"error at {content} ---- {input_token}")

            target_tokens.append(target_token)
            labels.append(label_ids)
            tokenized_inputs.input_ids[i] = input_ids
        tokenized_inputs["labels"] = target_tokens
        tokenized_inputs['ori_labels'] = labels
        return tokenized_inputs

    processed_raw_datasets = {}
    processed_raw_datasets['train'] = dataset['train'].map(
        tokenize_and_align_labels,
        batched=True,
        remove_columns=dataset["train"].column_names,
        desc="Running tokenizer on dataset",
    )

    label_words['i-val'] = [x for x in label_words['i-val'] if x not in keywords]
    for x in label_words['i-val']:
        if x in keywords:
            print(x)

    processed_raw_datasets['validation'] = dataset['validation'].map(
        tokenize_and_align_labels,
        batched=True,
        remove_columns=dataset["train"].column_names,
        desc="Running tokenizer on dataset",
    )

    return processed_raw_datasets, label_words, keywords
