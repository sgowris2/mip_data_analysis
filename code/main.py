import os
import pandas as pd
from pprint import pprint

import utils
from stats import run_stat_analysis


def run_analysis(filepath):
    df = pd.read_csv(filepath, dtype=object)
    df = df.drop_duplicates()
    df = utils.normalize_column_names(df)
    df = utils.set_null_strings_to_none(df)
    result = run_stat_analysis(df)
    pprint(result)


if __name__ == '__main__':
    directory = '../data'
    filename = 'mip_hr_2024-09.csv'
    run_analysis(os.path.join(directory, filename))