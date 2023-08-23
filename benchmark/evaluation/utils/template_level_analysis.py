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

from __future__ import print_function

import os
import pandas as pd
from evaluation.utils.common import is_abstract
from tqdm import tqdm


def evaluate_template_level(dataset, df_groundtruth, df_parsedresult, filter_templates=None):
    """
    Conduct the template-level analysis using 4-type classifications

    :param dataset:
    :param groundtruth:
    :param parsedresult:
    :param output_dir:
    :return: SM, OG, UG, MX
    """

    # oracle_templates = list(groundtruth_df['EventTemplate'].drop_duplicates().dropna())
    # identified_templates = list(parsedresult_df['EventTemplate'].drop_duplicates().dropna())
    correct_parsing_templates = 0
    if filter_templates is not None:
        filter_identify_templates = set()
    null_logids = df_groundtruth[~df_groundtruth['EventTemplate'].isnull()].index
    df_groundtruth = df_groundtruth.loc[null_logids]
    df_parsedresult = df_parsedresult.loc[null_logids]
    series_groundtruth = df_groundtruth['EventTemplate']
    series_parsedlog = df_parsedresult['EventTemplate']
    series_groundtruth_valuecounts = series_groundtruth.value_counts()
    # series_parsedlog_valuecounts = series_parsedlog.value_counts()

    df_combined = pd.concat([series_groundtruth, series_parsedlog], axis=1, keys=['groundtruth', 'parsedlog'])
    grouped_df = df_combined.groupby('parsedlog')
    
    for identified_template, group in tqdm(grouped_df):
    # for identified_template in tqdm(series_parsedlog_valuecounts.index):
        # Get the log_message_ids corresponding to the identified template from the tool-generated structured file
        # log_message_ids = parsedresult_df.loc[parsedresult_df['EventTemplate'] == identified_template, 'LineId']
        # log_message_ids = pd.DataFrame(log_message_ids)
        # log_message_ids = series_parsedlog[series_parsedlog == identified_template].index
        corr_oracle_templates = set(list(group['groundtruth']))
        # identify corr_oracle_templates using log_message_ids and oracle_structured_file
        # corr_oracle_templates = find_corr_oracle_templates(log_message_ids, groundtruth_df)
        if filter_templates is not None and len(corr_oracle_templates.intersection(set(filter_templates))) > 0:
            filter_identify_templates.add(identified_template)
        
        # print(set(identified_template), corr_oracle_templates)

        if corr_oracle_templates == {identified_template}:
            if (filter_templates is None) or (identified_template in filter_templates):
                correct_parsing_templates += 1
        # else:
            # print(corr_oracle_templates, set([identified_template]))

    if filter_templates is not None:
        PTA = correct_parsing_templates / len(filter_identify_templates)
        RTA = correct_parsing_templates / len(filter_templates)
    else:
        PTA = correct_parsing_templates / len(grouped_df)
        RTA = correct_parsing_templates / len(series_groundtruth_valuecounts)
    FTA = 0.0
    if PTA != 0 or RTA != 0:
        FTA = 2 * (PTA * RTA) / (PTA + RTA)
    print('PTA: {:.4f}, RTA: {:.4f} FTA: {:.4f}'.format(PTA, RTA, FTA))
    t1 = len(grouped_df) if filter_templates is None else len(filter_identify_templates)
    t2 = len(series_groundtruth_valuecounts) if filter_templates is None else len(filter_templates)
    print("Identify : {}, Groundtruth : {}".format(t1, t2))
    return t1, t2, FTA, PTA, RTA


def evaluate_template_level_lstm(dataset, df_groundtruth, df_parsedresult, filter_templates=None):
    """
    Conduct the template-level analysis using 4-type classifications

    :param dataset:
    :param groundtruth:
    :param parsedresult:
    :param output_dir:
    :return: SM, OG, UG, MX
    """

    # oracle_templates = list(groundtruth_df['EventTemplate'].drop_duplicates().dropna())
    # identified_templates = list(parsedresult_df['EventTemplate'].drop_duplicates().dropna())
    correct_parsing_templates = 0
    if filter_templates is not None:
        filter_identify_templates = set()
    null_logids = df_groundtruth[~df_groundtruth['EventTemplate'].isnull()].index
    df_groundtruth = df_groundtruth.loc[null_logids]
    df_parsedresult = df_parsedresult.loc[null_logids]
    series_groundtruth = df_groundtruth['EventTemplate']
    series_parsedlog = df_parsedresult['EventTemplate']
    series_groundtruth_valuecounts = series_groundtruth.value_counts()
    # series_parsedlog_valuecounts = series_parsedlog.value_counts()

    df_combined = pd.concat([series_groundtruth, series_parsedlog], axis=1, keys=['groundtruth', 'parsedlog'])
    grouped_df = df_combined.groupby('parsedlog')
    
    for identified_template, group in tqdm(grouped_df):
    # for identified_template in tqdm(series_parsedlog_valuecounts.index):
        # Get the log_message_ids corresponding to the identified template from the tool-generated structured file
        # log_message_ids = parsedresult_df.loc[parsedresult_df['EventTemplate'] == identified_template, 'LineId']
        # log_message_ids = pd.DataFrame(log_message_ids)
        # log_message_ids = series_parsedlog[series_parsedlog == identified_template].index
        corr_oracle_templates = set(list(group['groundtruth']))
        # identify corr_oracle_templates using log_message_ids and oracle_structured_file
        # corr_oracle_templates = find_corr_oracle_templates(log_message_ids, groundtruth_df)
        if filter_templates is not None and len(corr_oracle_templates.intersection(set(filter_templates))) > 0:
            filter_identify_templates.add(identified_template)
        
        # print(set(identified_template), corr_oracle_templates)

        if len(corr_oracle_templates) == 1 and correct_lstm(identified_template, list(corr_oracle_templates)[0]):
            if (filter_templates is None) or (list(corr_oracle_templates)[0] in filter_templates):
                correct_parsing_templates += 1
        # else:
            # print(corr_oracle_templates, set([identified_template]))

    print(correct_parsing_templates)
    print(len(grouped_df))
    print(len(series_groundtruth_valuecounts))
    if filter_templates is not None:
        PTA = correct_parsing_templates / len(filter_identify_templates)
        RTA = correct_parsing_templates / len(filter_templates)
    else:
        PTA = correct_parsing_templates / len(grouped_df)
        RTA = correct_parsing_templates / len(series_groundtruth_valuecounts)
    FTA = 0.0
    if PTA != 0 or RTA != 0:
        FTA = 2 * (PTA * RTA) / (PTA + RTA)
    print('PTA: {:.4f}, RTA: {:.4f} FTA: {:.4f}'.format(PTA, RTA, FTA))
    t1 = len(grouped_df) if filter_templates is None else len(filter_identify_templates)
    t2 = len(series_groundtruth_valuecounts) if filter_templates is None else len(filter_templates)
    print("Identify : {}, Groundtruth : {}".format(t1, t2))
    return t1, t2, FTA, PTA, RTA


def correct_lstm(groudtruth, parsedresult):
    tokens1 = groudtruth.split(' ')
    tokens2 = parsedresult.split(' ')
    tokens1 = ["<*>" if "<*>" in token else token for token in tokens1]
    return tokens1 == tokens2

# def find_corr_oracle_templates(log_message_ids, groundtruth_df):
#     """
#     Identify the corresponding oracle templates for the tool-generated(identified) template

#     :param log_message_ids: Log_message ids that corresponds to the tool-generated template
#     :param groundtruth_df: Oracle structured file
#     :return: Identified oracle templates that corresponds to the tool-generated(identified) template
#     """

#     corresponding_oracle_templates = groundtruth_df.merge(log_message_ids, on='LineId')
#     corresponding_oracle_templates = list(corresponding_oracle_templates.EventTemplate.unique())
#     return corresponding_oracle_templates


# def compute_template_level_accuracy(num_oracle_template, comparison_results_df):
#     """Calculate the template-level accuracy values.

#     :param num_oracle_template: total number of oracle templates
#     :param comparison_results_df: template analysis results (dataFrame)
#     :return: f1, precision, recall, over_generalized, under_generalized, mixed
#     """
#     count_total = float(len(comparison_results_df))
#     precision = len(comparison_results_df[comparison_results_df.type == 'SM']) / count_total  # PTA
#     recall = len(comparison_results_df[comparison_results_df.type == 'SM']) / float(num_oracle_template)  # RTA
#     over_generalized = len(comparison_results_df[comparison_results_df.type == 'OG']) / count_total
#     under_generalized = len(comparison_results_df[comparison_results_df.type == 'UG']) / count_total
#     mixed = len(comparison_results_df[comparison_results_df.type == 'MX']) / count_total
#     f1_measure = 0.0
#     if precision != 0 or recall != 0:
#         f1_measure = 2 * (precision * recall) / (precision + recall)
#     return f1_measure, precision, recall, over_generalized, under_generalized, mixed
