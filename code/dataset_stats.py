import pandas as pd


class DatasetStats:

    class ColumnStats:
        def __init__(self, column_name, count, unique, most_common):
            self.column_name = column_name
            self.count = count
            self.unique = unique
            self.most_common = most_common

        def __repr__(self):
            return {'column_name': self.column_name,
                    'count': self.count,
                    'unique': self.unique,
                    'most_common': self.most_common}

    def __init__(self, dataset_name: str, df: pd.DataFrame):
        self.dataset_name = dataset_name
        self.dataset_stats(df)

    def __repr__(self):
        result = {
            'no_of_unique_rows': self.no_of_unique_rows,
            'no_of_columns': self.no_of_columns,
            'column_names': self.column_names,
            'empty_columns': self.empty_columns,
            'column_stats': [x.__repr__() for x in self.column_stats]
        }
        return result

    def dataset_stats(self, df: pd.DataFrame):
        empty_columns = df.columns[df.isnull().all()]
        column_stats = df.describe(include='all')
        self.no_of_unique_rows = int(df.shape[0])
        self.no_of_columns = int(df.shape[1])
        self.column_names = df.columns.tolist()
        self.empty_columns = 'None' if len(empty_columns) == 0 else empty_columns
        self.column_stats = [self.ColumnStats(column_name=x,
                                              count=int(column_stats.loc['count'][x]),
                                              unique=int(column_stats.loc['unique'][x]),
                                              most_common=column_stats.loc['top'][x]) for x in column_stats]