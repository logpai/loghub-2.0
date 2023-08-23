import pandas as pd
import json
import regex as re


datasets = [
    "Proxifier",
    "Linux",
    "Apache",
    "Zookeeper",
    "Hadoop",
    "HealthApp",
    "OpenStack",
    "HPC",
    "Mac",
    "OpenSSH",
    "Spark",
    "Thunderbird",
    "BGL",
    "HDFS",
]


def modify_prompt(new_prompt, dataset):
    new_df = pd.read_csv(f"../../full_dataset/{dataset}/{dataset}_full.log_structured.csv")
    new_df = new_df.drop_duplicates(subset=['Content'])
    print(dataset, len(new_prompt))
    logs = [re.sub(' +', ' ', new_prompt[i]['text']).strip() for i in range(len(new_prompt))]
    new_df = new_df[new_df['Content'].isin(logs)]
    for i in range(len(new_prompt)):
        new_prompt[i]['text'] = re.sub(' +', ' ', new_prompt[i]['text']).strip()
        if len(new_df[new_df['Content'] == new_prompt[i]['text']]) > 0:
            new_prompt[i]['label'] = new_df[new_df['Content'] == new_prompt[i]['text']].iloc[0]['EventTemplate']
    return new_prompt


def process(dataset):
    new_prompt = []
    with open (f"datasets/{dataset}/{32}shot/1.json", "r") as fr:
        lines = fr.readlines()
        for line in lines:
            data = json.loads(line)
            new_prompt.append(data)
    new_prompt = modify_prompt(new_prompt, dataset)
    with open (f"datasets/{dataset}/{32}shot/2.json", "w") as fw:
        for data in new_prompt:
            fw.write(json.dumps(data) + "\n")


for dataset in datasets:
    process(dataset)


