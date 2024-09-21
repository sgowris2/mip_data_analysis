import json
import os
import pandas as pd
from pprint import pprint

import utils
import stats


def run_analysis(directory, filename):
    df = pd.read_csv(os.path.join(directory, filename), dtype=object)
    df = df.drop_duplicates()
    df = utils.normalize_column_names(df)
    df = utils.set_null_strings_to_none(df)
    dataset_stats = stats.dataset_stats(df)
    task_evidence_stats = stats.task_evidence_stats(df)

    pprint(dataset_stats)
    pprint(task_evidence_stats)
    with open('../output/dataset_stats__{}.json'.format(filename.replace('.csv', '')), 'w') as f:
        json.dump(dataset_stats, f)
    with open('../output/task_evidence_stats__{}.json'.format(filename.replace('.csv', '')), 'w') as f:
        json.dump(task_evidence_stats, f)


if __name__ == '__main__':
    directory = '../data'
    filename = 'mip_hr_2024-09.csv'
    run_analysis(directory, filename)