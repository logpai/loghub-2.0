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


def calculate_parsing_accuracy(groundtruth, parsedresult):

    parsedresult_df = pd.read_csv(parsedresult)
    groundtruth_df = pd.read_csv(groundtruth)

    correctly_parsed_messages = parsedresult_df[['EventTemplate']].eq(groundtruth_df[['EventTemplate']]).values.sum()
    total_messages = len(parsedresult_df[['Content']])

    PA = float(correctly_parsed_messages) / total_messages
    print('Parsing_Accuracy (PA): {:.4f}'.format(PA))

    return PA
