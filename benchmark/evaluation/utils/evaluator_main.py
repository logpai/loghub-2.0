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

import os
import time
import csv
import chardet
from multiprocessing import Process
from logparser.utils.evaluator import evaluate
from evaluation.utils.template_level_analysis import evaluate_template_level, evaluate_template_level_lstm
from evaluation.utils.PA_calculator import calculate_parsing_accuracy, calculate_parsing_accuracy_lstm
import pandas as pd

# TIMEOUT = 3600 * 12  # log template identification timeout (sec)
TIMEOUT = 3600 * 12  # log template identification timeout (sec)


def prepare_results(output_dir, otc, complex, frequent):
    if not os.path.exists(output_dir):
        # make output directory
        os.makedirs(output_dir)

    # make a new summary file
    result_file = 'summary_[otc={},complex={},frequent={}].csv'.format(str(otc), str(int(complex)), str(int(frequent)))
    with open(os.path.join(output_dir, result_file), 'w') as csv_file:
        fw = csv.writer(csv_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        # fw.writerow(['Dataset', 'GA_time', 'PA_time', 'TA_time', 'parse_time', 'identified_templates',
        #              'ground_templates', 'GA', 'PA', 'FTA', 'PTA', 'RTA', 'OG', 'UG', 'MX'])
        fw.writerow(['Dataset', 'parse_time', 'identified_templates',
                     'ground_templates', 'GA', 'PA', 'FGA', 'PTA', 'RTA', 'FTA'])

    return result_file


def is_file_empty(file_path):
    with open(file_path, 'r') as file:
        content = file.read()
        return len(content) == 0


def evaluator(
        dataset,
        input_dir,
        output_dir,
        log_file,
        LogParser,
        param_dict,
        otc,
        complex,
        frequent,
        result_file,
        lstm=False
):
    """
    Unit function to run the evaluation for a specific configuration.

    """

    print('\n=== Evaluation on %s ===' % dataset)
    indir = os.path.join(input_dir, os.path.dirname(log_file))
    log_file_basename = os.path.basename(log_file)
    if otc:
        # use a structured log file with corrected oracle templates
        groundtruth = os.path.join(indir, log_file_basename + '_structured_corrected.csv')
    else:
        groundtruth = os.path.join(indir, log_file_basename + '_structured.csv')

    parsedresult = os.path.join(output_dir, log_file_basename + '_structured.csv')
    # identify templates using Drain
    start_time = time.time()
    if LogParser != None:
        print("start parsing.")
        parser = LogParser(**param_dict)
        p = Process(target=parser.parse, args=(log_file_basename,))
        p.start()
        p.join(timeout=TIMEOUT)
        if p.is_alive():
            print('*** TIMEOUT for Template Identification')
            p.terminate()
            with open(parsedresult, 'w') as fw:
                pass
            return
        print("end parsing.")
        parse_time = time.time() - start_time  # end_time is the wall-clock time in seconds
    else:
        parse_time = -1
    print("parsing time: ", parse_time)

    # if not os.path.exists(parsedresult):
    #     with open(parsedresult, 'w') as fw:
    #         pass
    # print(parsedresult)
    if not os.path.exists(parsedresult) or is_file_empty(parsedresult):
        print("No output file generated.")
        result = dataset + ',' + \
                "None" + ',' + \
                "None" + ',' + \
                "None" + ',' + \
                "None" + ',' + \
                "None" + ',' + \
                "None" + ',' + \
                "None" + ',' + \
                "None" + ',' + \
                "None" + '\n'
                # "{:.1f}".format(GA_end_time) + ',' + \
                # "{:.1f}".format(PA_end_time) + ',' + \
                # "{:.1f}".format(TA_end_time) + ',' + \

        with open(os.path.join(output_dir, result_file), 'a') as summary_file:
            summary_file.write(result)
        return   


    filter_templates = None
    if complex != 0:
        print("Evaluate on complex mode: ", complex)
        template_file = os.path.join(indir, log_file_basename + '_templates.csv')
        df = pd.read_csv(template_file)
        if complex == 1:
            df = df[df['EventTemplate'].str.count('<*>') == 0]
        if complex == 2:
            df = df[(df['EventTemplate'].str.count('<*>') >= 1) & (df['EventTemplate'].str.count('<*>') <= 4)]
        if complex == 3:
            df = df[df['EventTemplate'].str.count('<*>') >= 5]
        filter_templates = df['EventTemplate'].tolist()
    if frequent != 0:
        print("Evaluate on frequent mode: ", frequent)
        template_file = os.path.join(indir, log_file_basename + '_templates.csv')
        df = pd.read_csv(template_file)
        df_sorted = df.sort_values('Occurrences')
        if frequent > 0:
            n = int(len(df_sorted) / 100.0 * frequent)
            filter_templates = df_sorted['EventTemplate'].tolist()[:n]
        else:
            n = len(df_sorted) - int(len(df_sorted) / 100.0 * -frequent)
            filter_templates = df_sorted['EventTemplate'].tolist()[n:]

    if filter_templates != None:
        print("length of filter templates: ", len(filter_templates))

    if filter_templates != None and len(filter_templates) == 0:
        # result = dataset + ',' + \
                # "None" + ',' + \
                # "None" + ',' + \
                # "None" + ',' + \
                # "None" + ',' + \
                # "None" + ',' + \
                # "None" + ',' + \
                # "None" + ',' + \
                # "None" + ',' + \
                # "None" + '\n'

        # with open(os.path.join(output_dir, result_file), 'a') as summary_file:
            # summary_file.write(result)
        return   

    parsedresult = pd.read_csv(parsedresult, dtype=str)
    parsedresult.fillna("", inplace=True)
    groundtruth = pd.read_csv(groundtruth, dtype=str)


    print("Start compute grouping accuracy")
    # calculate grouping accuracy
    start_time = time.time()
    GA, FGA = evaluate(groundtruth, parsedresult, filter_templates)

    GA_end_time = time.time() - start_time
    print('Grouping Accuracy calculation done. [Time taken: {:.3f}]'.format(GA_end_time))

    # calculate parsing accuracy
    start_time = time.time()
    if lstm == True:
        PA = calculate_parsing_accuracy_lstm(groundtruth, parsedresult, filter_templates)
        print("Finish calculate_parsing_accuracy_lstm")
    else:
        PA = calculate_parsing_accuracy(groundtruth, parsedresult, filter_templates)
    PA_end_time = time.time() - start_time
    print('Parsing Accuracy calculation done. [Time taken: {:.3f}]'.format(PA_end_time))

    # calculate template-level accuracy
    start_time = time.time()
    if lstm == True:
        tool_templates, ground_templates, FTA, PTA, RTA = evaluate_template_level_lstm(dataset, groundtruth, parsedresult, filter_templates)
    else:
        tool_templates, ground_templates, FTA, PTA, RTA = evaluate_template_level(dataset, groundtruth, parsedresult, filter_templates)
    TA_end_time = time.time() - start_time
    print('Template-level accuracy calculation done. [Time taken: {:.3f}]'.format(TA_end_time))

    result = dataset + ',' + \
             "{:.2f}".format(parse_time) + ',' + \
             str(tool_templates) + ',' + \
             str(ground_templates) + ',' + \
             "{:.3f}".format(GA) + ',' + \
             "{:.3f}".format(PA) + ',' + \
             "{:.3f}".format(FGA) + ',' + \
             "{:.3f}".format(PTA) + ',' + \
             "{:.3f}".format(RTA) + ',' + \
             "{:.3f}".format(FTA) + '\n'

    with open(os.path.join(output_dir, result_file), 'a') as summary_file:
        summary_file.write(result)
