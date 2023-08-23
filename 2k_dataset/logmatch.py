import os
import sys
sys.path.append('../')
import chardet
import pandas as pd
from logparser.logmatch import regexmatch
#from correct_postprocess import correct_single_template

datasets = [
    "Proxifier",
    "Linux",
    "Apache",
    "Zookeeper",
    "Mac",
    "OpenStack",
    "HealthApp",
    "Hadoop",
    "HPC",
    "OpenSSH",
    "Android",
    "BGL",
    "HDFS",
    "Spark",
    "Windows",
    "Thunderbird",
]
# datasets = [
#     "Thunderbird",
# ]
benchmark_settings = {
    'HDFS': {
        'log_file': 'HDFS/HDFS.log',
        'log_format': '<Date> <Time> <Pid> <Level> <Component>: <Content>',
        'regex': [r'blk_-?\d+', r'(\d+\.){3}\d+(:\d+)?'],
        'st': 0.5,
        'depth': 4
        },

    'Hadoop': {
        'log_file': 'Hadoop/Hadoop.log',
        'log_format': '<Date> <Time> <Level> \[<Process>\] <Component>: <Content>', 
        'regex': [r'(\d+\.){3}\d+'],
        'st': 0.5,
        'depth': 4        
        },

    'Spark': {
        'log_file': 'Spark/Spark.log',
        'log_format': '<Date> <Time> <Level> <Component>: <Content>', 
        'regex': [r'(\d+\.){3}\d+', r'\b[KGTM]?B\b', r'([\w-]+\.){2,}[\w-]+'],
        'st': 0.5,
        'depth': 4
        },

    'Zookeeper': {
        'log_file': 'Zookeeper/Zookeeper.log',
        'log_format': '<Date> <Time> - <Level>  \[<Node>:<Component>@<Id>\] - <Content>',
        'regex': [r'(/|)(\d+\.){3}\d+(:\d+)?'],
        'st': 0.5,
        'depth': 4        
        },

    'BGL': {
        'log_file': 'BGL/BGL.log',
        'log_format': '<Label> <Timestamp> <Date> <Node> <Time> <NodeRepeat> <Type> <Component> <Level> <Content>',
        'regex': [r'core\.\d+', r'\d+:[a-fA-F0-9]{8,}', r'[a-fA-F0-9]{8,}'], 
        'st': 0.6,
        'depth': 5
        },

    'HPC': {
        'log_file': 'HPC/HPC.log',
        'log_format': '<LogId> <Node> <Component> <State> <Time> <Flag> <Content>',
        'regex': [r'=\d+'],
        'st': 0.5,
        'depth': 4
        },

    'Thunderbird': {
        'log_file': 'Thunderbird/Thunderbird.log',
        'log_format': '<Label> <Timestamp> <Date> <User> <Month> <Day> <Time> <Location> <Component>(\[<PID>\])?: <Content>',
        'regex': [r'(\d+\.){3}\d+'],
        'st': 0.5,
        'depth': 4        
        },

    'Windows': {
        'log_file': 'Windows/Windows.log',
        'log_format': '<Date> <Time>, <Level>                  <Component>    <Content>',
        'regex': [r'0x.*?\s'],
        'st': 0.7,
        'depth': 5      
        },

    'Linux': {
        'log_file': 'Linux/Linux.log',
        'log_format': '<Month> <Date> <Time> <Level> <Component>(\[<PID>\])?: <Content>',
        'regex': [r'(\d+\.){3}\d+', r'\d{2}:\d{2}:\d{2}'],
        'st': 0.39,
        'depth': 6        
        },

    'Android': {
        'log_file': 'Android/Android.log',
        'log_format': '<Date> <Time>  <Pid>  <Tid> <Level> <Component>: <Content>',
        'regex': [r'(/[\w-]+)+', r'([\w-]+\.){2,}[\w-]+', r'\b(\-?\+?\d+)\b|\b0[Xx][a-fA-F\d]+\b|\b[a-fA-F\d]{4,}\b'],
        'st': 0.2,
        'depth': 6   
        },

    'HealthApp': {
        'log_file': 'HealthApp/HealthApp.log',
        'log_format': '<Time>\|<Component>\|<Pid>\|<Content>',
        'regex': [r'\d+##\d+##\d+##\d+##\d+##\d+', r'=\d+'],
        'st': 0.2,
        'depth': 4
        },

    'Apache': {
        'log_file': 'Apache/Apache.log',
        'log_format': '\[<Time>\] \[<Level>\] <Content>',
        'regex': [r'(\d+\.){3}\d+'],
        'st': 0.5,
        'depth': 4        
        },

    'Proxifier': {
        'log_file': 'Proxifier/Proxifier.log',
        'log_format': '\[<Time>\] <Program> - <Content>',
        'regex': [r'<\d+\ssec', r'([\w-]+\.)+[\w-]+(:\d+)?', r'\d{2}:\d{2}(:\d{2})*', r'[KGTM]B'],
        'st': 0.6,
        'depth': 3
        },

    'OpenSSH': {
        'log_file': 'OpenSSH/OpenSSH.log',
        'log_format': '<Date> <Day> <Time> <Component> sshd\[<Pid>\]: <Content>',
        'regex': [r'(\d+\.){3}\d+', r'([\w-]+\.){2,}[\w-]+'],
        'st': 0.6,
        'depth': 5   
        },

    'OpenStack': {
        'log_file': 'OpenStack/OpenStack.log',
        'log_format': '<Logrecord> <Date> <Time> <Pid> <Level> <Component> \[<ADDR>\] <Content>',
        'regex': [r'\[instance:\s*(.*?)\]', r'((\d+\.){3}\d+,?)+', r'/.+?\s', r'\d+'],
        'st': 0.2,
        'depth': 4
        },

    'Mac': {
        'log_file': 'Mac/Mac.log',
        'log_format': '<Month>  <Date> <Time> <User> <Component>\[<PID>\]( \(<Address>\))?: <Content>',
        'regex': [r'([\w-]+\.){2,}[\w-]+'],
        'st': 0.7,
        'depth': 6   
        },
}


def custom_sort(string):
    cleaned_string = string.replace('<*>', '').replace(' ', '')
    length = len(cleaned_string)
    asterisk_count = string.count('<*>')
                        
    return (-length, asterisk_count)


def process_csv(input_file, output_file, encoding="utf-8"):
    df = pd.read_csv(input_file, encoding=encoding)

    print(df.columns)
    label_column = df.columns[3]
    print("Label columns: ", label_column)
    df = df.dropna(subset=[label_column])
    data = df[label_column].tolist()

    clean_label = set()
    for d in data:
        lines = d.split('\n')
        for line in lines:
            if line.strip() != "":
                clean_label.add(line.strip())

    clean_label = list(clean_label)
    clean_label = sorted(clean_label, key=custom_sort)

    new_df = pd.DataFrame(clean_label, columns=["EventTemplate"])
    new_df.insert(0, "EventId", [f"E{i}" for i in range(1, 1 + len(new_df))])
    new_df['EventTemplate'] = new_df['EventTemplate'].apply(correct_single_template)
    new_df.to_csv(output_file, index=False, encoding="utf-8")
    print("Preprocessed template CSV: ", len(new_df))
    

def clean_structured_file(file_path):
    print("===============================================================")
    print("### clean structred file:")
    output_path = file_path.replace("_structured", "_structured_cleaned")
    df = pd.read_csv(file_path)
    df = df.drop(['ParameterList'], axis=1)
    print("Original length: ", len(df))
    df = df[df["Content"].notna() & (df["Content"] != 'NONE')]
    df = df[df["EventId"].notna() & (df["EventId"] != 'NONE')]
    df = df[df["EventTemplate"].notna() & (df["EventTemplate"] != 'NONE')]
    df.to_csv(output_path, index=None, encoding="utf-8")
    print("Processed length: ", len(df))
    print("===============================================================")
    return output_path


def clean_template_file(file_path):
    print("===============================================================")
    print("### clean template file:")
    output_path = file_path.replace("_templates", "_templates_cleaned")
    df = pd.read_csv(file_path)
    print("Original length: ", len(df))
    df = df[df["EventId"].notna() & (df["EventId"] != 'NONE')]
    df = df[df["EventTemplate"].notna() & (df["EventTemplate"] != 'NONE')]
    df = df[df["Occurrences"].notna() & (df["Occurrences"] != 'NONE') & (df["Occurrences"] > 0)]
    df.to_csv(output_path, index=None, encoding="utf-8")
    print("Processed length: ", len(df))
    print("===============================================================")
    return output_path


def clean_log_file(csv_path, log_path):
    print("===============================================================")
    print("### clean log file:")
    csv_data = pd.read_csv(csv_path, index_col=None)
    line_ids = csv_data['LineId'].tolist()

    with open(log_path, "r") as f:
        lines = f.readlines()
    print("Original length: ", len(lines))
    new_log_path = os.path.join("../../full_dataset/", log_path.replace(".log", "_full.log"))
    print(new_log_path)
    cnt = 0
    with open(new_log_path, "w") as f:
        for line_id in line_ids:
            if line_id <= len(lines):
                cnt += 1
                f.write(lines[line_id - 1])
    print("Processed length: ", cnt)
    print("===============================================================")
    return new_log_path


def diff_templates(final_templates, original_templates):
    print("===============================================================")
    print("Start diff templates")
    diff_templates_path = original_templates.replace("original.csv", "diff.csv")
    df1 = pd.read_csv(final_templates)
    df2 = pd.read_csv(original_templates)
    diff = set(df2['EventTemplate']) - set(df1['EventTemplate'])
    
    df_diff = pd.DataFrame({'EventTemplate': list(diff)})
    if len(df_diff) > 0:
        df_diff.to_csv(diff_templates_path, index=False, encoding="utf-8")
    print("End diff templates: ", len(df_diff))
    print("===============================================================")



def process_full_log(dataset):
    with open(f"{dataset}/{dataset}.log_templates_label.csv", "rb") as f:
            result = chardet.detect(f.read())
    process_csv(f"{dataset}/{dataset}.log_templates_label.csv", f"{dataset}/{dataset}.log_templates_original.csv", result['encoding'])
    output_dir   = f"logmatch_result/{dataset}" # The result directory
    full_output_dir = f"../../full_dataset/{dataset}"
    if not os.path.exists(full_output_dir):
        os.makedirs(full_output_dir)
    log_filepath = benchmark_settings[dataset]['log_file'] # The input log file path
    log_format   = benchmark_settings[dataset]['log_format'] # HDFS log format
    n_workers    = 32 # The number of workers in parallel
    template_filepath = log_filepath + '_templates_original.csv' # The event template file path
    csv_filepath = log_filepath + '_structured.csv'
    matcher = regexmatch.PatternMatch(outdir=output_dir, n_workers=n_workers, logformat=log_format, optimized=True)
    matcher.match(log_filepath, template_filepath, "")
    # diff_templates(log_filepath + "_templates.csv", template_filepath)
    
    output_csv = os.path.join("logmatch_result/", csv_filepath)
    output_template = os.path.join("logmatch_result/", log_filepath + "_templates.csv")
    
    cleaned_structred_path = clean_structured_file(output_csv)
    cleaned_template_path = clean_template_file(output_template)
    full_log_path = clean_log_file(cleaned_structred_path, log_filepath)
    
    full_matcher = regexmatch.PatternMatch(outdir=full_output_dir, n_workers=n_workers, logformat=log_format, optimized=True)
    full_matcher.match(full_log_path, cleaned_template_path)
    df = pd.read_csv(full_log_path + "_structured.csv")
    df = df.drop(["ParameterList"], axis=1)
    df.to_csv(full_log_path + "_structured_all.csv", index=None, encoding="utf-8")

    df = df[['LineId', 'Content', 'EventId', 'EventTemplate']]
    df.to_csv(full_log_path + "_structured.csv", index=None, encoding="utf-8")

    diff_templates(full_log_path + "_templates.csv", template_filepath)
    



if __name__ == "__main__":
    for dataset in datasets:
        log_format   = benchmark_settings[dataset]['log_format']
        output_dir = f"{dataset}"
        log_filepath = f"{dataset}/{dataset}_2k.log"
        template_filepath = f"{dataset}/{dataset}_2k.log_templates_corrected.csv"
        matcher = regexmatch.PatternMatch(outdir=output_dir, n_workers=8, logformat=log_format, optimized=True)
        matcher.match(log_filepath, template_filepath, "")
        print(dataset)
        # process_full_log(dataset)
