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

import pandas as pd
from logppt.evaluation.utils.common import is_abstract


def evaluate_template_level(groundtruth, parsedresult):
    """
    Conduct the template-level analysis using 4-type classifications

    :param dataset:
    :param groundtruth:
    :param parsedresult:
    :param output_dir:
    :return: SM, OG, UG, MX
    """

    oracle_templates = list(pd.read_csv(groundtruth)['EventTemplate'].drop_duplicates().dropna())
    identified_templates = list(pd.read_csv(parsedresult)['EventTemplate'].drop_duplicates().dropna())
    parsedresult_df = pd.read_csv(parsedresult)
    groundtruth_df = pd.read_csv(groundtruth)
    # groundtruth_df['EventTemplate'] = groundtruth_df['EventTemplate'].str.lower()
    comparison_results = []
    for identified_template in identified_templates:
        identified_template_type = None

        # Get the log_message_ids corresponding to the identified template from the tool-generated structured file
        log_message_ids = parsedresult_df.loc[parsedresult_df['EventTemplate'] == identified_template, 'LineId']
        log_message_ids = pd.DataFrame(log_message_ids)
        num_messages = len(log_message_ids)

        # identify corr_oracle_templates using log_message_ids and oracle_structured_file
        corr_oracle_templates = find_corr_oracle_templates(log_message_ids, groundtruth_df)

        # Check SM (SaMe)
        if set(corr_oracle_templates) == {identified_template}:
            identified_template_type = 'SM'

        # incorrect template analysis
        if identified_template_type is None:

            # determine the template type
            template_types = set()
            for corr_oracle_template in corr_oracle_templates:
                if is_abstract(identified_template, corr_oracle_template):
                    template_types.add('OG')
                elif is_abstract(corr_oracle_template, identified_template):
                    template_types.add('UG')
                else:
                    template_types.add('MX')

            if len(template_types) == 1:  # if the set is singleton
                identified_template_type = template_types.pop()
            else:
                identified_template_type = 'MX'

        # save the results for the current identified template
        comparison_results.append([identified_template, identified_template_type, corr_oracle_templates, num_messages])

    comparison_results_df = pd.DataFrame(comparison_results,
                                         columns=['identified_template', 'type', 'corr_oracle_templates', 'num_messages'])
    # comparison_results_df.to_csv(os.path.join(output_dir, dataset + '_template_analysis_results.csv'), index=False)
    (F1_measure, PTA, RTA, OG, UG, MX) = compute_template_level_accuracy(len(oracle_templates), comparison_results_df)
    print('F1: {:.4f}, PTA: {:.4f}, RTA: {:.4f}, OG: {:.4f}, UG: {:.4f}, MX: {:.4f}'.format(F1_measure, PTA, RTA, OG, UG, MX))
    return len(identified_templates), len(oracle_templates), F1_measure, PTA, RTA, OG, UG, MX


def find_corr_oracle_templates(log_message_ids, groundtruth_df):
    """
    Identify the corresponding oracle templates for the tool-generated(identified) template

    :param log_message_ids: Log_message ids that corresponds to the tool-generated template
    :param groundtruth_df: Oracle structured file
    :return: Identified oracle templates that corresponds to the tool-generated(identified) template
    """

    corresponding_oracle_templates = groundtruth_df.merge(log_message_ids, on='LineId')
    corresponding_oracle_templates = list(corresponding_oracle_templates.EventTemplate.unique())
    return corresponding_oracle_templates


def compute_template_level_accuracy(num_oracle_template, comparison_results_df):
    """Calculate the template-level accuracy values.

    :param num_oracle_template: total number of oracle templates
    :param comparison_results_df: template analysis results (dataFrame)
    :return: f1, precision, recall, over_generalized, under_generalized, mixed
    """
    count_total = float(len(comparison_results_df))
    precision = len(comparison_results_df[comparison_results_df.type == 'SM']) / count_total  # PTA
    recall = len(comparison_results_df[comparison_results_df.type == 'SM']) / float(num_oracle_template)  # RTA
    over_generalized = len(comparison_results_df[comparison_results_df.type == 'OG']) / count_total
    under_generalized = len(comparison_results_df[comparison_results_df.type == 'UG']) / count_total
    mixed = len(comparison_results_df[comparison_results_df.type == 'MX']) / count_total
    f1_measure = 0.0
    if precision != 0 or recall != 0:
        f1_measure = 2 * (precision * recall) / (precision + recall)
    return f1_measure, precision, recall, over_generalized, under_generalized, mixed
