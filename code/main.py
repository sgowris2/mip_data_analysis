import json
from json2table import convert
import os
import pandas as pd

from dataset_stats import DatasetStats
import utils
from task_analysis import TaskAnalysis


def run_analysis(data: pd.DataFrame, title, output_dir, show_plots, save_plots):

    dataset_stats = DatasetStats(dataset_name=title, df=data)
    dataset_stats_html = convert(dataset_stats.__repr__(), table_attributes={'border': 1})

    task_analysis = TaskAnalysis(dataset_name='{}_overall'.format(title), df=data)
    task_evidence_stats_html = convert(task_analysis.__repr__(), table_attributes={'border': 1})

    task_analysis.visualize(dataset_name=title, output_dir=output_dir, show_plots=show_plots, save_plots=save_plots)

    with open(os.path.join(output_dir, '{}__dataset_stats.html'.format(title)), 'w') as f:
        f.write(dataset_stats_html)
    with open(os.path.join(output_dir, '{}__task_evidence_stats.html'.format(title)), 'w') as f:
        f.write(task_evidence_stats_html)


def load_and_normalize_file(directory, filename):
    d = pd.read_csv(os.path.join(directory, filename), dtype=object)
    d = d.drop_duplicates()
    d = utils.normalize_column_names(d)
    d = utils.set_null_strings_to_none(d)
    d = utils.normalize_column_data(d, ['project_title'])
    return d


if __name__ == '__main__':
    dir = '../data/Haryana 2024'
    filenames = os.listdir(dir)

    for fn in filenames:
        if fn.endswith('.csv'):
            print('\n\n-------------------------\nAnalyzing file: {}'.format(fn))
            df = load_and_normalize_file(dir, fn)
            run_analysis(df, title=fn.replace('.csv', '').replace(' ', '').strip(),
                         output_dir='../output/Haryana 2024',
                         show_plots=False, save_plots=True)
            print('------------------------------------------\n------------------------------------------\n')
