import pandas as pd


class DatasetStats:

    def __init__(self, dataset_name: str, df: pd.DataFrame):
        self.dataset_name = dataset_name
        self.dataset_stats(df)

    def __repr__(self):
        result = {
            'no_of_unique_rows': self.no_of_unique_rows,
            'no_of_columns': self.no_of_columns,
            'column_names': self.column_names,
            'empty_columns': self.empty_columns,
            'column_stats': self.column_stats
        }
        return result

    def dataset_stats(self, df: pd.DataFrame):
        empty_columns = list(df.columns[df.isnull().all()].values)
        column_stats = df.describe(include='all')
        self.no_of_unique_rows = int(df.shape[0])
        self.no_of_columns = int(df.shape[1])
        self.column_names = df.columns.tolist()
        self.empty_columns = 'None' if len(empty_columns) == 0 else empty_columns
        self.column_stats = column_stats.reset_index()