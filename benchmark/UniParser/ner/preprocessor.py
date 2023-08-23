import json
import re
import random
# from spacy.lang.id import Indonesian
import spacy


def create_json_labels():
    files = {
        "train": [
            "../data/raw/ner1/data_train.txt",
            "../data/raw/ner2/training_data.txt"
        ],
        "test": [
            "../data/raw/ner1/data_test.txt",
            "../data/raw/ner2/testing_data.txt",
        ]
    }
    for corpus in files:
        annotations = []
        for file in files[corpus]:
            with open(file, "r") as f:
                texts = f.readlines()
            for text in texts:
                annotation = {
                    "text": "",
                    "labels": []
                }
                text = text.split("\t")[0]
                text = re.sub(r"\s+", " ", text)
                text = re.sub(r"<ENAMEX TYPE=\"([A-Z]+)\">", r"<\1>", text)
                for element in text.strip().split("<"):
                    if len(element) > 0 and element[0] == "/":
                        tag, rest = element.split(">")
                        annotation["text"] += rest
                    elif ">" in element:
                        tag, content = element.split(">")
                        content = content.strip()
                        annotation["labels"].append(
                            (len(annotation["text"]), len(annotation["text"]) + len(content), tag)
                        )
                        annotation["text"] += content
                    else:
                        annotation["text"] += element
                annotations.append(annotation)
        with open(f"../data/processed/{corpus}/{corpus}.json", "w") as f:
            json.dump(annotations, f, indent=2)
        print(f"[{corpus}] Processed {len(annotations)} sentences.")


def transform_json_to_conll():
    # nlp = Indonesian()
    nlp = spacy.load("en_core_web_sm")
    file = "../data/processed/test/test.json"
    with open(file, "r") as f:
        annotations = json.load(f)
    random.seed(1339)
    random.shuffle(annotations)
    buffer_conll = {
        "val": "",
        "test": ""
    }
    for anno_i, annotation in enumerate(annotations):
        sorted_labels = sorted(annotation["labels"], key=lambda label: (label[0], label[1]))
        token_i = 0
        curr_label = sorted_labels[token_i] if len(sorted_labels) > 0 else None
        tokens = nlp(annotation["text"])
        for token in tokens:
            token_begin = token.idx
            token_end = token.idx + len(token.text)
            tag = "O"
            if curr_label and token_begin >= curr_label[0] and token_end <= curr_label[1]:
                tag = curr_label[2]
                if token_end == curr_label[1]:
                    tag = f"L-{tag}" if token_begin > curr_label[0] else f"U-{tag}"
                    if token_i < len(sorted_labels) - 1:
                        token_i += 1
                        curr_label = sorted_labels[token_i]
                elif token_begin == curr_label[0]:
                    tag = f"B-{tag}"
                else:
                    tag = f"I-{tag}"
            buffer_conll["val" if anno_i <= len(annotations) // 2 else "test"] += token.text + "\t" + tag + "\n"
        buffer_conll["val" if anno_i <= len(annotations) // 2 else "test"] += "\n"
    with open("../input/val.tsv", "w") as f:
        f.write(buffer_conll["val"])
    with open("../input/test.tsv", "w") as f:
        f.write(buffer_conll["test"])


if __name__ == "__main__":
    transform_json_to_conll()
