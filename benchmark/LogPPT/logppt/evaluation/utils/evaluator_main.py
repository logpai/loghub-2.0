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

from logppt.evaluation.evaluator import evaluate
from logppt.evaluation.utils.template_level_analysis import evaluate_template_level

TIMEOUT = 3600  # log template identification timeout (sec)


def prepare_results(output_dir, otc):
    if not os.path.exists(output_dir):
        # make output directory
        os.makedirs(output_dir)

    # make a new summary file
    result_file = 'summary_[otc={}].csv'.format(str(otc))
    with open(os.path.join(output_dir, result_file), 'w') as csv_file:
        fw = csv.writer(csv_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        fw.writerow(['Dataset', 'GA_time', 'PA_time', 'TA_time', 'parse_time', 'identified_templates',
                     'ground_templates', 'GA', 'PA', 'FTA', 'PTA', 'RTA', 'OG', 'UG', 'MX'])

    return result_file


def evaluator(groundtruth, parsedresult):
    """
    Unit function to run the evaluation for a specific configuration.

    """
    # calculate grouping accuracy
    start_time = time.time()
    GA, PA, ED, _, unseen_PA, no_unseen = evaluate(
        groundtruth=groundtruth,
        parsedresult=parsedresult
    )
    GA_end_time = time.time() - start_time
    print('Grouping and Parsing Accuracy calculation done. [Time taken: {:.3f}]'.format(GA_end_time))
    print(f"Unseen Accuracy {unseen_PA} - {no_unseen}")
    # calculate template-level accuracy
    start_time = time.time()
    tool_templates, ground_templates, FTA, PTA, RTA, OG, UG, MX = evaluate_template_level(
        groundtruth=groundtruth,
        parsedresult=parsedresult
    )
    TA_end_time = time.time() - start_time
    print('Template-level accuracy calculation done. [Time taken: {:.3f}]'.format(TA_end_time))

    return GA, PA, ED, FTA, PTA, RTA, OG, UG, MX, unseen_PA, no_unseen
