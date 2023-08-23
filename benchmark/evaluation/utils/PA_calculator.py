"""
This file is part of TA-Eval-Rep.
Copyright (C) 2022 University of Luxembourg
    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, version 3 of the License.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import pandas as pd
import regex as re


def post_process_tokens(tokens, punc):
    excluded_str = ['=', '|', '(', ')']
    for i in range(len(tokens)):
        if tokens[i].find("<*>") != -1:
            tokens[i] = "<*>"
        else:
            new_str = ""
            for s in tokens[i]:
                if (s not in punc and s != ' ') or s in excluded_str:
                    new_str += s
            tokens[i] = new_str
    return tokens


def message_split(message):
    punc = "!\"#$%&'()+,-/:;=?@.[\]^_`{|}~"
    splitters = "\s\\" + "\\".join(punc)
    splitter_regex = re.compile("([{}]+)".format(splitters))
    tokens = re.split(splitter_regex, message)
    tokens = list(filter(lambda x: x != "", tokens))
    
    #print("tokens: ", tokens)
    tokens = post_process_tokens(tokens, punc)
    tokens = [
        token.strip()
        for token in tokens
        if token != "" and token != ' ' 
    ]
    tokens = [
        token
        for idx, token in enumerate(tokens)
        if not (token == "<*>" and idx > 0 and tokens[idx - 1] == "<*>")
    ]
    return tokens


def calculate_similarity(template1, template2):
    template1 = message_split(template1)
    template2 = message_split(template2)
    intersection = len(set(template1).intersection(set(template2)))
    union = (len(template1) + len(template2)) - intersection
    return intersection / union


def calculate_parsing_accuracy(groundtruth_df, parsedresult_df, filter_templates=None):

    # parsedresult_df = pd.read_csv(parsedresult)
    # groundtruth_df = pd.read_csv(groundtruth)
    if filter_templates is not None:
        groundtruth_df = groundtruth_df[groundtruth_df['EventTemplate'].isin(filter_templates)]
        parsedresult_df = parsedresult_df.loc[groundtruth_df.index]
    correctly_parsed_messages = parsedresult_df[['EventTemplate']].eq(groundtruth_df[['EventTemplate']]).values.sum()
    total_messages = len(parsedresult_df[['Content']])

    PA = float(correctly_parsed_messages) / total_messages

    # similarities = []
    # for index in range(len(groundtruth_df)):
    #     similarities.append(calculate_similarity(groundtruth_df['EventTemplate'][index], parsedresult_df['EventTemplate'][index]))
    # SA = sum(similarities) / len(similarities)
    # print('Parsing_Accuracy (PA): {:.4f}, Similarity_Accuracy (SA): {:.4f}'.format(PA, SA))
    print('Parsing_Accuracy (PA): {:.4f}'.format(PA))
    return PA


def calculate_parsing_accuracy_lstm(groundtruth_df, parsedresult_df, filter_templates=None):

    # parsedresult_df = pd.read_csv(parsedresult)
    # groundtruth_df = pd.read_csv(groundtruth)
    if filter_templates is not None:
        groundtruth_df = groundtruth_df[groundtruth_df['EventTemplate'].isin(filter_templates)]
        parsedresult_df = parsedresult_df.loc[groundtruth_df.index]
    # correctly_parsed_messages = parsedresult_df[['EventTemplate']].eq(groundtruth_df[['EventTemplate']]).values.sum()
    groundtruth_templates = list(groundtruth_df['EventTemplate'])
    parsedresult_templates = list(parsedresult_df['EventTemplate'])
    correctly_parsed_messages = 0
    for i in range(len(groundtruth_templates)):
        if correct_lstm(groundtruth_templates[i], parsedresult_templates[i]):
            correctly_parsed_messages += 1

    PA = float(correctly_parsed_messages) / len(groundtruth_templates)

    # similarities = []
    # for index in range(len(groundtruth_df)):
    #     similarities.append(calculate_similarity(groundtruth_df['EventTemplate'][index], parsedresult_df['EventTemplate'][index]))
    # SA = sum(similarities) / len(similarities)
    # print('Parsing_Accuracy (PA): {:.4f}, Similarity_Accuracy (SA): {:.4f}'.format(PA, SA))
    print('Parsing_Accuracy (PA): {:.4f}'.format(PA))
    return PA


def correct_lstm(groudtruth, parsedresult):
    tokens1 = groudtruth.split(' ')
    tokens2 = parsedresult.split(' ')
    tokens1 = ["<*>" if "<*>" in token else token for token in tokens1]
    return tokens1 == tokens2