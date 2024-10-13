import pandas as pd
import numpy as np


def normalize_column_data(df: pd.DataFrame, columns: list):
    for col in columns:
        df[col] = df[col].replace('\'', '').replace('"', '')
    return df


def set_null_strings_to_none(df: pd.DataFrame):
    df = df.replace('Null', np.nan)
    df = df.replace('null', np.nan)
    return df


def normalize_column_names(df: pd.DataFrame):
    column_mapper = {x: _normalize_string(x) for x in df.columns.values}
    df.rename(columns=column_mapper, inplace=True)
    return df


def _normalize_string(s):
    s = s.strip().replace(' ', '_')
    s = str.lower(s)
    return s