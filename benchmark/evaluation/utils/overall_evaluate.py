import pandas as pd
import scipy.special
from nltk.metrics.distance import edit_distance
from sklearn.metrics import accuracy_score
import numpy as np
import os
from IPython import embed

def evaluate(dataset, output_path, groundtruth, parsedresult, result_path):
    df_groundtruth = pd.read_csv(groundtruth)
    df_parsedlog = pd.read_csv(parsedresult, index_col=False)
    # df_groundtruth['EventTemplate'] = df_groundtruth['EventTemplate'].str.lower()
    #embed()
    # Remove invalid groundtruth event Ids
    null_logids = df_groundtruth[~df_groundtruth['EventTemplate'].isnull()].index
    
    df_groundtruth = df_groundtruth.loc[null_logids]
    df_parsedlog = df_parsedlog.loc[null_logids]
    
    
    GA, FGA = get_group_accuracy(df_groundtruth['EventTemplate'], df_parsedlog['EventTemplate'])

    correctly_parsed_messages = df_parsedlog[['EventTemplate']].eq(df_groundtruth[['EventTemplate']]).values.sum()
    total_messages = len(df_parsedlog[['Content']])

    PA = float(correctly_parsed_messages) / total_messages

    # edit_distance_result = []
    # for i, j in zip(np.array(df_groundtruth.EventTemplate.values, dtype='str'),
    #                 np.array(df_parsedlog.EventTemplate.values, dtype='str')):
    #     edit_distance_result.append(edit_distance(i, j))
    # edit_distance_result_mean = np.mean(edit_distance_result)
    # edit_distance_result_std = np.std(edit_distance_result)

    
    tool_templates, ground_templates, FTA, PTA, RTA = evaluate_template_level(
        dataset=dataset,
        groundtruth=groundtruth,
        parsedresult=parsedresult,
        output_dir=output_path
    )

    # tool_templates, ground_templates, FTA, PTA, RTA, OG, UG, MX = evaluate_template_level_all(
    #     dataset=dataset,
    #     groundtruth=groundtruth,
    #     parsedresult=parsedresult,
    #     output_dir=output_path
    # )

    result = dataset + ',' + \
             str(tool_templates) + ',' + \
             str(ground_templates) + ',' + \
             "{:.3f}".format(GA) + ',' + \
             "{:.3f}".format(FGA) + ',' + \
             "{:.3f}".format(PA) + ',' + \
             "{:.3f}".format(FTA) + ',' + \
             "{:.3f}".format(PTA) + ',' + \
             "{:.3f}".format(RTA) + '\n'
                 
    print(result)

    with open(result_path, 'a') as summary_file:
        summary_file.write(result)
    # print(
    #     'Precision: %.4f, Recall: %.4f, F1_measure: %.4f, Group Accuracy: %.4f, Message-Level Accuracy: %.4f, Edit Distance: %.4f' % (
    #         precision, recall, f_measure, GA, PA, edit_distance_result_mean))

    #return GA, PA


def get_group_accuracy(series_groundtruth, series_parsedlog, debug=False):
    series_parsedlog_valuecounts = series_parsedlog.value_counts()
    accurate_events = 0  # determine how many lines are correctly parsed
    accurate_templates = 0
    #print(series_parsedlog_valuecounts)
    for parsed_eventId in series_parsedlog_valuecounts.index:
        # print(parsed_eventId)
        logIds = series_parsedlog[series_parsedlog == parsed_eventId].index
        series_groundtruth_logId_valuecounts = series_groundtruth[logIds].value_counts()
        error_eventIds = (parsed_eventId, series_groundtruth_logId_valuecounts.index.tolist())
        error = True
        if series_groundtruth_logId_valuecounts.size == 1:
            groundtruth_eventId = series_groundtruth_logId_valuecounts.index[0]
            if logIds.size == series_groundtruth[series_groundtruth == groundtruth_eventId].size:
                accurate_events += logIds.size
                accurate_templates += 1
                error = False
        if error and debug:
            print('(parsed_eventId, groundtruth_eventId) =', error_eventIds, 'failed', logIds.size, 'messages')

    GA = float(accurate_events) / series_groundtruth.size
    FGA = float(accurate_templates) / len(series_groundtruth.value_counts())
    
    return GA, FGA


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


def evaluate_template_level(dataset, groundtruth, parsedresult, output_dir):
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

    correct_parsing_templates = 0
    for identified_template in identified_templates:

        # Get the log_message_ids corresponding to the identified template from the tool-generated structured file
        log_message_ids = parsedresult_df.loc[parsedresult_df['EventTemplate'] == identified_template, 'LineId']
        log_message_ids = pd.DataFrame(log_message_ids)

        # identify corr_oracle_templates using log_message_ids and oracle_structured_file
        corr_oracle_templates = find_corr_oracle_templates(log_message_ids, groundtruth_df)

        # Check SM (SaMe)
        # print("==========================================")
        # print("Oracle: ", corr_oracle_templates)
        # print("Identify: ", identified_template)
        # print(set(corr_oracle_templates) == {identified_template})
        # print("==========================================")
        if set(corr_oracle_templates) == {identified_template}:
            correct_parsing_templates += 1
            
    PTA = correct_parsing_templates / len(identified_templates)
    RTA = correct_parsing_templates / len(oracle_templates)
    FTA = 0.0
    if PTA != 0 or RTA != 0:
        FTA = 2 * (PTA * RTA) / (PTA + RTA)
        
    return len(identified_templates), len(oracle_templates), FTA, PTA, RTA


# ================================================================================


def is_abstract(x, y):
    """
    Determine if template_x is more abstract than template_y.

    :param x: a template (str)
    :param y: a template or a message (str)
    :return: True if x is more abstract (general) than y
    """

    if y is np.nan:
        return False

    m = re.match(get_pattern_from_template(x), y)
    if m:
        return True
    else:
        return False


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



def evaluate_template_level_all(dataset, groundtruth, parsedresult, output_dir):
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
    comparison_results_df.to_csv(os.path.join(output_dir, dataset + '_template_analysis_results.csv'), index=False)
    (F1_measure, PTA, RTA, OG, UG, MX) = compute_template_level_accuracy(len(oracle_templates), comparison_results_df)
    print('F1: {:.4f}, PTA: {:.4f}, RTA: {:.4f}, OG: {:.4f}, UG: {:.4f}, MX: {:.4f}'.format(F1_measure, PTA, RTA, OG, UG, MX))
    return len(identified_templates), len(oracle_templates), F1_measure, PTA, RTA, OG, UG, MX
