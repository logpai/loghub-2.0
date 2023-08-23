from __future__ import print_function

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

import time
import regex as re
import os
import pandas as pd
import numpy as np
import argparse
from datetime import datetime
from natsort import natsorted


datasets = ['HDFS', 'Hadoop', 'Spark', 'Zookeeper', 'OpenStack', 'BGL', 'HPC', 'Thunderbird', 'Windows', 'Linux',
            'Mac', 'Android', 'HealthApp', 'Apache', 'OpenSSH', 'Proxifier']  # to keep the order of the systems


def sort_templates(templates):
    """
    Sort templates by its length. The shorter, the later.

    :param templates: a list of templates
    :return: a sorted list of templates
    """
    return sorted(templates, key=lambda x: len(x), reverse=True)


def get_pattern_from_template(template):
    escaped = re.escape(template)
    spaced_escape = re.sub(r'\\\s+', "\\\s+", escaped)
    return "^" + spaced_escape.replace(r"<\*>", r"(\S+?)") + "$"  # a single <*> can consume multiple tokens


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


def common_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-otc', '--oracle_template_correction',
                        help="Set this if you want to use corrected oracle templates",
                        default=False, action='store_true')
    args = parser.parse_args()
    return args


def unique_output_dir(name):
    return os.path.join('{}_result'.format(name), '{}_{}'.format(datetime.now().strftime("%Y%m%d_%H%M%S"), os.getpid()))


def correct_single_template(template, user_strings=None):
    """Apply all rules to process a template.

    DS (Double Space)
    BL (Boolean)
    US (User String)
    DG (Digit)
    PS (Path-like String)
    WV (Word concatenated with Variable)
    DV (Dot-separated Variables)
    CV (Consecutive Variables)

    """

    boolean = {'true', 'false'}
    default_strings = {'null', 'root', 'admin'}
    path_delimiters = {  # reduced set of delimiters for tokenizing for checking the path-like strings
        r'\s', r'\,', r'\!', r'\;', r'\:',
        r'\=', r'\|', r'\"', r'\'',
        r'\[', r'\]', r'\(', r'\)', r'\{', r'\}'
    }
    token_delimiters = path_delimiters.union({  # all delimiters for tokenizing the remaining rules
        r'\.', r'\-', r'\+', r'\@', r'\#', r'\$', r'\%', r'\&',
    })

    if user_strings:
        default_strings = default_strings.union(user_strings)

    # apply DS
    template = template.strip()
    template = re.sub(r'\s+', ' ', template)

    # apply PS
    p_tokens = re.split('('+'|'.join(path_delimiters)+')', template)
    new_p_tokens = []
    for p_token in p_tokens:
        if re.match(r'^(\/[^\/]+)+$', p_token):
            p_token = '<*>'
        new_p_tokens.append(p_token)
    template = ''.join(new_p_tokens)

    # tokenize for the remaining rules
    tokens = re.split('('+'|'.join(token_delimiters)+')', template)  # tokenizing while keeping delimiters
    new_tokens = []
    for token in tokens:
        # apply BL, US
        for to_replace in boolean.union(default_strings):
            if token.lower() == to_replace.lower():
                token = '<*>'

        # apply DG
        if re.match(r'^\d+$', token):
            token = '<*>'

        # apply WV
        if re.match(r'^[^\s\/]*<\*>[^\s\/]*$', token):
            if token != '<*>/<*>':  # need to check this because `/` is not a deliminator
                token = '<*>'

        # collect the result
        new_tokens.append(token)

    # make the template using new_tokens
    template = ''.join(new_tokens)

    # Substitute consecutive variables only if separated with any delimiter including "." (DV)
    while True:
        prev = template
        template = re.sub(r'<\*>\.<\*>', '<*>', template)
        if prev == template:
            break

    # Substitute consecutive variables only if not separated with any delimiter including space (CV)
    # NOTE: this should be done at the end
    while True:
        prev = template
        template = re.sub(r'<\*><\*>', '<*>', template)
        if prev == template:
            break

    return template


def correct_templates_and_update_files(dir_path, log_file_basename, inplace=False):
    """
    Wrapper function to update the structured log and templates file, mainly for (1) EventId and (2) EventTemplate.
    When a template is revised, update 'EventTemplate' in the structured log whose EventId is matching.
    When multiple templates become one via PP, update both 'EventId' and 'EventTemplate' in the structured log
    whose EventId is matching with multiple (original) templates.
    Note that, in the latter case, update 'EventId' as the first EventId of the original templates, so that
    the same template has the same EventId in the resulting structured log.

    Example:
        <original templates (EventId, EventTemplate)>
        E1, Send 123
        E2, Send 456

        <original structured log (lineID, message, EventId, EventTemplate)>
        1, Send 123, E1, Send 123
        2, Send 456, E2, Send 456

        <templates after PP (EventId, EventTemplate)>
        E1, Send <*>

        <expected structured log (lineID, message, EventId, EventTemplate)>
        1, Send 123, E1, Send <*>
        2, Send 456, E1, Send <*>

    :param dir_path: a directory to write the structured log and template files
    :param log_file_basename: a base file name (ex: BGL_2k.log)
    :param inplace: whether the update is performed on the existing file or not
    :return: None (internally update `structured_log_file` and `templates_file`)
    """

    # identify org_structured_log_file
    org_structured_log_file = os.path.join(dir_path, log_file_basename + '_structured.csv')

    # read original structured log
    structured_logs_df = pd.read_csv(org_structured_log_file)

    # extract templates_df
    # dropna is required because LogSig generates empty template for Hadoop
    templates_df = structured_logs_df.drop_duplicates(subset='EventTemplate').dropna(subset=['EventTemplate'])

    # convert templates_df into dict (key: EventId, value: EventTemplate)
    templates_dict = templates_df.set_index('EventId')['EventTemplate'].to_dict()

    # correct templates using 8 rules
    new_templates_dict = correct_templates(templates_dict)

    # update structured_logs_df
    start_time = time.time()
    for index, row in structured_logs_df.iterrows():

        # try to search matching template using 'EventId'
        is_matched = False
        for tids, template in new_templates_dict.items():
            if row['EventId'] in tids:
                structured_logs_df.at[index, 'EventId'] = tids[0]
                structured_logs_df.at[index, 'EventTemplate'] = template
                is_matched = True
                break

        # if no matching template, report it
        if is_matched is False:
            print('*** WARN: No matching template; EventId:', row['EventId'], 'message:', row['Content'])

    # update structured log file
    if inplace:  # used for tool-generated structured log file
        structured_logs_df.to_csv(org_structured_log_file, index=False)
    else:  # used for oracle structured log file
        structured_logs_df.to_csv(os.path.join(dir_path, log_file_basename + '_structured_corrected.csv'), index=False)

    # update templates file
    new_templates = []
    for tids, template in new_templates_dict.items():
        new_templates.append((tids[0], template))
    new_templates = natsorted(new_templates, key=lambda x: x[0])
    new_templates_df = pd.DataFrame(new_templates, columns=['EventId', 'EventTemplate'])
    new_templates_df.to_csv(os.path.join(dir_path, log_file_basename + '_templates_corrected.csv'), index=False)

    print('Structured log and templates file update done. [Time taken: {:.3f}]'.format(time.time() - start_time))


def correct_templates(templates_dict):
    """
    Core function for template postprocessing.

    :param templates_dict: existing templates (key: EventId, value: EventTemplate)
    :return: new templates_dict (key: Tuple of EventIds, value: EventTemplate)
    """

    # templates that are affected by the post-processing
    change_count = 0
    inverse_templates_dict = {}  # key: EventTemplate, value: list of EventIds

    start_time = time.time()
    for tid, template in sorted(templates_dict.items(), key=lambda x: x[0]):  # sort to avoid non-determinism
        org_template = template
        new_template = correct_single_template(template)

        # count the number of changed templates
        if org_template != new_template:
            change_count += 1

        # update temp_templates_dict
        if new_template in inverse_templates_dict.keys():
            inverse_templates_dict[new_template].append(tid)
        else:
            inverse_templates_dict[new_template] = [tid]

    # build new_templates_dict using inverse_templates_dict
    new_templates_dict = {tuple(tids): template for template, tids in inverse_templates_dict.items()}

    end_time = time.time() - start_time
    print('\tOriginal templates:', len(templates_dict.keys()))
    print('\tTemplates after correction:', len(new_templates_dict.keys()))
    print("\tTemplates changed by correction:", change_count)
    print('Template correction done. [Time taken: {:.3f}]'.format(end_time))

    return new_templates_dict
