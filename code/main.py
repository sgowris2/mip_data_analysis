import json
from json2table import convert
import os
import pandas as pd

import utils
import stats


def run_analysis(directory, filename, create_plots, show_plots, save_plots):
    df = pd.read_csv(os.path.join(directory, filename), dtype=object)
    df = df.drop_duplicates()
    df = utils.normalize_column_names(df)
    df = utils.set_null_strings_to_none(df)
    df = utils.normalize_column_data(df, ['project_title'])
    dataset_stats = stats.dataset_stats(df)
    task_evidence_stats = stats.task_evidence_stats(df,
                                                    plot_title_string='{}_overall'.format(filename.replace('.csv', '').strip()),
                                                    create_plots=create_plots,
                                                    output_dir='../output',
                                                    show_plots=show_plots,
                                                    save_plots=save_plots)

    dataset_stats_html = convert(dataset_stats, table_attributes={'border': 1})
    task_evidence_stats_html = convert(task_evidence_stats, table_attributes={'border': 1})

    with open('../output/dataset_stats__{}.html'.format(filename.replace('.csv', '')), 'w') as f:
        f.write(dataset_stats_html)
    with open('../output/task_evidence_stats__{}.html'.format(filename.replace('.csv', '')), 'w') as f:
        f.write(task_evidence_stats_html)


if __name__ == '__main__':
    directory = '../data/Haryana 2024'
    filenames = os.listdir(directory)

    for filename in filenames:
        if filename.endswith('.csv'):
            run_analysis(directory, filename, create_plots=True, show_plots=False, save_plots=True)

    # run_analysis(directory, 'Quadrilaterals.csv', create_plots=True)