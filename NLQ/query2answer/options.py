"""
Reading command line options.
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import argparse


def read_command_line():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        required=True,
        help="path to the file where the queries to answer are stored",
    )
    parser.add_argument(
        "--clips_path",
        required=True,
        help="path where the clips .mp4 files are stored"
    )
    parser.add_argument(
        "--output_path",
        required=True,
        help="path where the answers to the queries will be stored"
    )
    configs = parser.parse_args()
    return configs
