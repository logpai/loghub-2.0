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
from evaluation.utils.common import correct_templates_and_update_files, datasets


def main():
    for system in datasets:
        print('-' * 70)
        print(system)
        dir_path = os.path.join('..', 'logs', system)
        log_file_basename = system + '_2k.log'
        correct_templates_and_update_files(dir_path, log_file_basename, inplace=False)


if __name__ == '__main__':
    main()

