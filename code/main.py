import json
from json2table import convert
import os
import pandas as pd

import utils
import stats


def run_analysis(df, title, create_plots, show_plots, save_plots):
    dataset_stats = stats.dataset_stats(df)
    task_evidence_stats = stats.task_evidence_stats(df,
                                                    plot_title_string='{}_overall'.format(title),
                                                    create_plots=create_plots,
                                                    output_dir='../output',
                                                    show_plots=show_plots,
                                                    save_plots=save_plots)

    dataset_stats_html = convert(dataset_stats, table_attributes={'border': 1})
    task_evidence_stats_html = convert(task_evidence_stats, table_attributes={'border': 1})

    with open('../output/dataset_stats__{}.html'.format(title), 'w') as f:
        f.write(dataset_stats_html)
    with open('../output/task_evidence_stats__{}.html'.format(title), 'w') as f:
        f.write(task_evidence_stats_html)


def load_and_normalize_file(directory, filename):
    df = pd.read_csv(os.path.join(directory, filename), dtype=object)
    df = df.drop_duplicates()
    df = utils.normalize_column_names(df)
    df = utils.set_null_strings_to_none(df)
    df = utils.normalize_column_data(df, ['project_title'])
    return df


if __name__ == '__main__':
    directory = '../data/Haryana 2024'
    filenames = os.listdir(directory)

    # combined_df = None
    # for filename in filenames:
    #     df = load_and_normalize_file(directory, filename)
    #     if combined_df is None:
    #         combined_df = df
    #     else:
    #         combined_df = pd.concat([combined_df, df], ignore_index=True)
    # run_analysis(combined_df, title='combined', create_plots=False, show_plots=False, save_plots=False)

    for filename in filenames:
        if filename.endswith('.csv'):
            df = load_and_normalize_file(directory, filename)
            run_analysis(df, title=filename.replace('.csv', '').strip(),
                         create_plots=True, show_plots=False, save_plots=True)