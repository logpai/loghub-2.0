from datasets import load_dataset


def load_data_parsing(main_args):
    # Get the datasets: the data file are JSON files
    data_files = {}
    if main_args.train_file is not None:
        data_files["train"] = [main_args.train_file]
    if main_args.unlabeled_file is not None:
        data_files["train"].append(main_args.unlabeled_file)
    if main_args.validation_file is not None:
        data_files["validation"] = main_args.validation_file
    if main_args.dev_file is not None:
        data_files["dev"] = main_args.dev_file

    raw_datasets = load_dataset("json", data_files=data_files)

    if raw_datasets["train"] is not None:
        column_names = raw_datasets["train"].column_names
    else:
        column_names = raw_datasets["validation"].column_names

    if main_args.text_column_name is not None:
        text_column_name = main_args.text_column_name
    elif "tokens" in column_names:
        text_column_name = "tokens"
    else:
        text_column_name = column_names[0]

    if main_args.label_column_name is not None:
        label_column_name = main_args.label_column_name
    else:
        label_column_name = column_names[1]

    return raw_datasets, text_column_name, label_column_name


def load_data_anomaly_detection(main_args):
    # Get the datasets: the data file are JSON files
    data_files = {}
    if main_args.train_file is not None:
        data_files["train"] = main_args.train_file
    # if main_args.validation_file is not None:
    #     data_files["validation"] = main_args.validation_file
    # if main_args.dev_file is not None:
    #     data_files["dev"] = main_args.dev_file

    raw_datasets = load_dataset("json", data_files=data_files)

    if raw_datasets["train"] is not None:
        column_names = raw_datasets["train"].column_names
    else:
        column_names = raw_datasets["validation"].column_names

    if main_args.text_column_name is not None:
        text_column_name = main_args.text_column_name
    elif "tokens" in column_names:
        text_column_name = "tokens"
    else:
        text_column_name = column_names[0]

    if main_args.label_column_name is not None:
        label_column_name = main_args.label_column_name
    else:
        label_column_name = column_names[1]

    return raw_datasets, text_column_name, label_column_name
