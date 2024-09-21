import pandas as pd
import numpy as np


def run_stat_analysis(df: pd.DataFrame):

    empty_columns = df.columns[df.isnull().all()]
    column_stats = df.describe(include='all')
    column_stats.rename(index={'count': 'total_no_of_values',
                               'unique': 'no_of_unique_values',
                               'top': 'most_common_value',
                               'freq': 'no_of_instances_of_most_common_value'}, inplace=True)

    result = {
        'no_of_unique_rows': df.shape[0],
        'no_of_columns': df.shape[1],
        'column_names': df.columns.values,
        'empty_columns': 'None' if len(empty_columns) == 0 else empty_columns,
        'column_stats': [{x: dict(column_stats[x])} for x in column_stats],
    }
    print(result)
    return result
