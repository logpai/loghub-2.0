import os
from collections import Counter
import json
import argparse
import regex as re
import pandas as pd

def write_json(data, path):
    with open(path, 'w', encoding='utf-8') as fp:
        json.dump(data, fp, indent=1)
def read_json(path):
    with open(path, 'r', encoding='utf-8') as fp:
        data = json.load(fp)
    return data


def write_ner_tsv_data(data, path):
    with open(path, 'w', encoding='utf-8') as fout:
        for _pair in data:
            lines = "\n".join([f"{i}\t{j}" for i, j in _pair])
            fout.write(lines + '\n\n')



#####################################################
#####################################################
######## Use Parsing CSV file #######################
#####################################################
import csv
def csv_reader(path):
    with open(path, 'r', encoding='utf-8') as fp:
        reader = csv.reader(fp)
        data = [i for i in reader]
    return data
def csv_writer(path, header, data):
    with open(path, 'w', encoding='utf-8_sig', newline="") as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(header)
        writer.writerows(data)

def modify_prompt(logs, templates, dataset):
    df = pd.read_csv(f"../../full_dataset/{dataset}/{dataset}_full.log_structured.csv")
    df = df.drop_duplicates(subset=['Content'])
    logs = [re.sub(' +', ' ', log).strip() for log in logs]
    df = df[df['Content'].isin(logs)]
    unknown_cnt = 0
    for i in range(len(logs)):
        if len(df[df['Content'] == logs[i]]) > 0:
            templates[i] = df[df['Content'] == logs[i]].iloc[0]['EventTemplate']
        else:
            unknown_cnt += 1
    print(unknown_cnt)
    return logs, templates

datasets = ['Apache', 'BGL', 'HDFS', 'HPC', 'Hadoop', 'HealthApp', 'Linux', 'Mac', 'OpenSSH', 'OpenStack',
         'Proxifier', 'Spark', 'Thunderbird', 'Zookeeper']

config = argparse.ArgumentParser()
config.add_argument('-full', '--full_data',
                    help="Set this if you want to test on full dataset",
                    default=False, action='store_true')
config = config.parse_args()
data_type = 'full' if config.full_data else '2k'
sentenceList = []
for dataset in datasets:
    print(f"{dataset}")
    data_dir = f'../../2k_dataset/{dataset}'
    data = csv_reader(os.path.join(data_dir, f"{dataset}_2k.log_structured_corrected.csv"))
    header = data[0]
    contents = data[1:]
    logs_idx = header.index("Content")
    templates_idx = header.index("EventTemplate")
    logs, templates = [i[logs_idx] for i in contents], [i[templates_idx] for i in contents]
    if config.full_data:
        logs, templates = modify_prompt(logs, templates, dataset)
    output_dir = f"{data_type}_annotations/{dataset}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    pairs = []
    all_ner_labels = Counter()
    for _log, _template in zip(logs, templates):
        tokens_log, tokens_template = _log.split(' '), _template.split(' ')
        label = ["B" if "<*>" in tok else "O" for tok in tokens_template]
        if len(tokens_log) != len(tokens_template) or len(tokens_log) != len(label):
            continue
        _pair = [[i, j] for i, j in zip(tokens_log, label)]
        pairs.append(_pair)
        all_ner_labels.update([i[1] for i in _pair])
    print(f"{dataset}: {all_ner_labels}")
    print(f"{dataset} # pairs: {len(pairs)}")
    # write the pairs into files (ner labels)
    write_ner_tsv_data(pairs, os.path.join(output_dir, 'logpub_parsing.txt'))
    write_json(all_ner_labels, os.path.join(output_dir, "logpub_lables.json"))

    # write the pairs into files (Binary label)
    import random
    random.seed(42)
    train = random.sample(list(range(len(pairs))), int(len(pairs) * 0.2))
    dev = random.sample([i for i in range(len(pairs)) if i not in train], int(len(pairs) * 0.2))
    test = [i for i in range(len(pairs)) if i not in train and i not in dev]
    if not os.path.exists(os.path.join(output_dir, 'logpub_bin_random')):
        os.makedirs(os.path.join(output_dir, 'logpub_bin_random'))
    write_ner_tsv_data([pairs[i] for i in train], os.path.join(output_dir, 'logpub_bin_random', 'train.tsv'))
    write_ner_tsv_data([pairs[i] for i in dev], os.path.join(output_dir, 'logpub_bin_random', 'val.tsv'))
    write_ner_tsv_data([pairs[i] for i in test], os.path.join(output_dir, 'logpub_bin_random', 'test.tsv'))



"""
Hadoop: Counter({'O': 12870, 'B': 3860})
Hadoop # pairs: 2000


"""

