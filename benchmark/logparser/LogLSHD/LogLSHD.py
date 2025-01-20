import hashlib
import os
from collections import Counter
from datetime import datetime

import numpy as np
import pandas as pd
import regex as re
from datasketch import MinHash, MinHashLSH
from fastdtw import fastdtw
from tqdm import tqdm

from .DTW import DTW_TemplateGenerator
    
class LogParser():
    def __init__(self, indir, log_format, outdir, rex=[], k=1, sig_len=20, cls_rate=0.7):
        """
        Initialize LogParser with LSH.
        """
        self.path = indir
        self.savepath = outdir
        self.log_format = log_format
        self.rex = rex
        self.rex_pattern = re.compile("|".join(self.rex))
        self.word_pattern = re.compile(r'^[a-zA-Z]+[.,]*$')  # Define a regular expression pattern to match pure words (only letters)

        self.k = k  # k-gram shingle
        self.sig_len = sig_len  # Length of signature
        self.cls_rate = cls_rate

        # template generation paramrters
        self.repr_count = 100 # from a group
        self.word_freq_filter = False
        self.freq_sample_size = 100
        self.is_DTW = True
        
        if self.is_DTW:
            self.DTW = DTW_TemplateGenerator()

        self.template_dict = {}

    def parse(self, filename):
        print(f"LSH: k={self.k},sig_len={self.sig_len},cls_rate={self.cls_rate}")

        start_time = datetime.now()
        filepath = os.path.join(self.path, filename)
        self.filename = filename

        print('Parsing file: ' + filepath)
        headers, regex = self.generate_logformat_regex(self.log_format)
        self.df_log = self.log_to_dataframe(filepath, regex, headers, self.log_format)
        self.df_log["EventTemplate"] = None
        self.df_log["EventId"] = None
        
        self.df_log['TokenCount'] = self.df_log['Content'].apply(lambda x: x.count(' '))
        self.df_log['ContentLength'] = self.df_log['Content'].apply(len)
        
        self.df_log['Char1']= self.df_log['Content'].apply(lambda x: x[0] if len(x) > 0 else '')
        self.df_log['Char2'] = self.df_log['Content'].apply(lambda x: x[max(0, len(x) // 4 - 1)] if len(x) >= 4 else '') # 25
        self.df_log['Char3'] = self.df_log['Content'].apply(lambda x: x[max(0, len(x) // 2 - 1)] if len(x) >= 2 else '') # 50
        self.df_log['Char4'] = self.df_log['Content'].apply(lambda x: x[max(0, 3 * len(x) // 4 - 1)] if len(x) >= 4 else '') # 75
        self.df_log['Char5'] = self.df_log['Content'].apply(lambda x: x[-1] if len(x) > 0 else '') # 100

        grouped_logs = self.df_log.groupby(['TokenCount','ContentLength','Char1','Char2','Char3','Char4','Char5'])
        grouped_keys = list(grouped_logs.groups.keys())
        # print('Group Size: ', len(grouped_keys))

        self.df_log.drop(['TokenCount','ContentLength','Char1','Char2','Char3','Char4','Char5'], axis=1, inplace=True)

        if self.cls_rate == 1:
            for i, (_, group) in enumerate(grouped_logs):
                # Find the template for the cluster
                group_idxs = group.index.tolist()
                template = self.find_template(group_idxs)

                template_updates = [] 
                eventid_updates = []

                # Generate a unique EventId based on the template using MD5 hash
                if template not in self.template_dict:
                    event_id = hashlib.md5(template.encode('utf-8')).hexdigest()[0:8]
                    self.template_dict[template] = event_id
                else:
                    event_id = self.template_dict[template]
                # Update the DataFrame with the new templates and event IDs
                for log_idx in group_idxs:
                    template_updates.append((log_idx, template))
                    eventid_updates.append((log_idx, event_id))
            
                # Update the DataFrame with the generated templates and EventId
                if template_updates:
                    template_indices, template_values = zip(*template_updates)
                    eventid_indices, eventid_values = zip(*eventid_updates)
                    self.df_log.loc[list(template_indices), 'EventTemplate'] = list(template_values)
                    self.df_log.loc[list(eventid_indices), 'EventId'] = list(eventid_values)
        else:

            lsh = MinHashLSH(threshold=self.cls_rate, num_perm=self.sig_len)
            print('LSH number of bands: ', lsh.b, ', band size: ', lsh.r)

            # Insert representative signatures into LSH
            minhashes = []
            hash_representations = []
            for i, (_, group) in enumerate(grouped_logs):
                # repr_shingle = group.iloc[0]['Shingle']
                repr_shingle = self.build_token_shingle(group.iloc[0]['Content'], self.k)
                minhash = MinHash(num_perm=self.sig_len)
                for element in repr_shingle:
                    minhash.update(str(element).encode('utf8'))

                lsh.insert(str(i), minhash)  # Insert the minhash with index i as identifier
                minhashes.append(minhash)
                hash_representations.append(tuple(minhash.hashvalues))
            
                
            visited = set()
            for i, minhash in tqdm(enumerate(minhashes), desc="Merge Clusters by LSH Similarity"):
                # print(visited)
                if i in visited:
                    # print('pass ',i)
                    continue

                # Query LSH to find clusters similar to the current one
                similar_clusters = lsh.query(minhashes[i])
                # print('add', similar_clusters)
                
                # repr_idxs = []
                merged_clusters = []
                for idx in similar_clusters:
                    similar_group = grouped_logs.get_group(grouped_keys[int(idx)]).index.tolist() 
                    # repr_idxs.extend(similar_group[:self.repr_count])
                    merged_clusters.extend(similar_group)
                    visited.add(int(idx))

                # Find the template for the cluster
                template = self.find_template(merged_clusters)

                template_updates = [] 
                eventid_updates = []

                # Generate a unique EventId based on the template using MD5 hash
                if template not in self.template_dict:
                    event_id = hashlib.md5(template.encode('utf-8')).hexdigest()[0:8]
                    self.template_dict[template] = event_id
                else:
                    event_id = self.template_dict[template]
                # Update the DataFrame with the new templates and event IDs
                for log_index in merged_clusters:
                    template_updates.append((log_index, template))
                    eventid_updates.append((log_index, event_id))
            
                # Update the DataFrame with the generated templates and EventId
                if template_updates:
                    template_indices, template_values = zip(*template_updates)
                    eventid_indices, eventid_values = zip(*eventid_updates)
                    self.df_log.loc[list(template_indices), 'EventTemplate'] = list(template_values)
                    self.df_log.loc[list(eventid_indices), 'EventId'] = list(eventid_values)

        self.wirteResultToFile()
        print('Parsing done. [Time taken: {!s}]'.format(datetime.now() - start_time))
    
    def build_token_shingle(self, log: str, k: int):
        """
        Split a log message into k-token shingle (Token-level shingle).
        """   
        # print('log:', log)
        # Split the content into tokens and keep only pure word tokens
        tokens = [token for token in log.split() if self.word_pattern.match(token)]
        # tokens = [token for token in log.split()]

        # log = self.rex_pattern.sub('', log)
        # tokens = log.split()

        if not tokens:
            return set([log])
        
        # Generate k-token shingles
        shingles = []
        for i in range(len(tokens) - k + 1):
            shingles.append(' '.join(tokens[i:i + k]))
        
        # If no shingles were generated, add the whole content as a single shingle
        if not shingles:
            shingles.append(' '.join(tokens))
        # print('shingles: ',shingles)

        return set(shingles)
    
    
    def find_template(self, log_idxs, sample_size=10):

        selected_idxs = []

        if self.word_freq_filter:
            # Randomly select 100 log from this cluster
            if len(log_idxs) > self.freq_sample_size:
                sampled_indices_for_freq = np.random.choice(log_idxs, self.freq_sample_size, replace=False)
            else:
                sampled_indices_for_freq = log_idxs

            # Calculate token frequencies in the sampled logs
            all_tokens = [
                token for idx in sampled_indices_for_freq
                for token in self.df_log.iloc[idx]['Content'].split()
            ]
            token_counts = Counter(all_tokens)

            # Select top 10 high-frequency tokens
            top_tokens = {token for token, _ in token_counts.most_common(10)}

            # Use these top tokens to pick the most representative samples
            selected_idxs = sorted(
                log_idxs,
                key=lambda idx: sum(token_counts[token] for token in self.preprocess(self.df_log.iloc[idx]['Content']).split() if token in top_tokens),
                reverse=True
            )[:sample_size]
        else:
            # If the cluster size is smaller than the sample size, use the whole cluster
            if len(log_idxs) <= sample_size:
                selected_idxs = log_idxs
            else:
                # Randomly sample a subset of indices from the cluster
                selected_idxs = np.random.choice(log_idxs, sample_size, replace=False)
            
        if self.is_DTW:

            selected_logs = [
                self.preprocess(self.df_log.iloc[idx]['Content'])
                for idx in selected_idxs
            ]
            # for log in selected_logs:
            #     print(log)

            # Find the common parts using DTW
            common_part = self.DTW.dynamic_time_warping(selected_logs)
            # print('common_part: ', common_part)
            # Use the common part to substitute non-matching portions in the original string
            substituted_string = self.DTW.combine_consecutive_star(common_part)
            # print(substituted_string)
            template = self.DTW.postprocess(substituted_string)
            # print(substituted_string)
            return template
    
        else:

            sampled_tokens = [
                self.preprocess(self.df_log.iloc[idx]['Content']).split()
                for idx in selected_idxs
            ]

            # for sample in sampled_tokens:
            #     print(sample)
        
            # Step 1: Choose the first sampled log as the base template
            base_template = sampled_tokens[0]

            # Initialize a list to store the modified template
            final_template = list(base_template)

            # print(final_template)

            # Step 2: Iterate over the remaining sampled tokens and compare with the base template
            for tokens in sampled_tokens[1:]:
                for token_idx in range(len(final_template)):
                    # Only compare if the token exists in the current log (lengths may differ)
                    if token_idx < len(tokens):
                        # If the current token differs from the base template, replace it with "<*>"
                        if tokens[token_idx] != final_template[token_idx]:
                            final_template[token_idx] = "<*>"
                    else:
                        # If the current log is shorter than the base template, mark the remaining tokens as "<*>"
                        final_template[token_idx] = "<*>"

            # Step 3: Merge consecutive <*> into a single one
            merged_template = []
            for token in final_template:
                if token == "<*>":
                    if len(merged_template) == 0 or merged_template[-1] != "<*>":
                        merged_template.append(token)
                else:
                    merged_template.append(token)
            
            return ' '.join(merged_template)

    
    def preprocess(self, line):
        """
        Preprocess a log line based on the provided regular expressions (rex).
        This replaces certain patterns with <*>.
        """
        for currentRex in self.rex:
            line = re.sub(currentRex, '<*>', line)
        return line

    def get_template(self, content_i, content_j):
        """
        Generate an event template by comparing two log lines.
        """
        words_i = content_i.split()
        words_j = content_j.split()
        template = []
        
        for w1, w2 in zip(words_i, words_j):
            if w1 == w2:
                template.append(w1)  # Keep static parts
            else:
                template.append("<*>")  # Replace variable parts with <*>
        
        return " ".join(template)

    def wirteResultToFile(self):
        """
        Output results to two files: structured log file and template file.
        """
        
        # Ensure the result directory exists
        if not os.path.isdir(self.savepath):
            os.makedirs(self.savepath)

        self.df_log.to_csv(os.path.join(self.savepath, self.filename + '_structured.csv'), index=False, quoting=1, escapechar='\\')
        print(f"Structured log written to {os.path.join(self.savepath, self.filename + '_structured.csv')}")

        # Count occurrences of each EventTemplate
        occ_dict = dict(self.df_log['EventTemplate'].value_counts())
        
        # Create a DataFrame for event templates
        df_event = pd.DataFrame()
        df_event['EventTemplate'] = self.df_log['EventTemplate'].unique()
        df_event['Occurrences'] = df_event['EventTemplate'].map(occ_dict)
        

        # Generate EventId using MD5 hash
        df_event['EventId'] = df_event['EventTemplate'].map(lambda x: hashlib.md5(x.encode('utf-8')).hexdigest()[0:8])

        # Re-arrange columns to match the desired format and save to CSV
        df_event = df_event[['EventId', 'EventTemplate', 'Occurrences']]
        df_event.to_csv(os.path.join(self.savepath, self.filename + '_templates.csv'), index=False, quoting=1, escapechar='\\')
        print(f"Template log written to {os.path.join(self.savepath, self.filename + '_templates.csv')}")
    
    def log_to_dataframe(self, log_file, regex, headers, logformat):
        """
        Transform log file to dataframe.
        """
        log_messages = []
        linecount = 0
        with open(log_file, 'r') as fin:
            for line in tqdm(fin, desc="Processing log lines"):
                try:
                    match = regex.search(line.strip())
                    message = [match.group(header) for header in headers]
                    log_messages.append(message)
                    linecount += 1
                except Exception as e:
                    pass
        logdf = pd.DataFrame(log_messages, columns=headers)
        logdf.insert(0, 'LineId', [i + 1 for i in range(linecount)])  # LineId for original order tracking
        return logdf

    def generate_logformat_regex(self, logformat):
        """
        Generate regular expression to split log messages.
        """
        headers = []
        splitters = re.split(r'(<[^<>]+>)', logformat)
        regex = ''
        for k in range(len(splitters)):
            if k % 2 == 0:
                splitter = re.sub(' +', '\\s+', splitters[k])
                regex += splitter
            else:
                header = splitters[k].strip('<').strip('>')
                regex += '(?P<%s>.*?)' % header
                headers.append(header)
        regex = re.compile('^' + regex + '$')
        return headers, regex
    