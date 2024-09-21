import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def dataset_stats(df: pd.DataFrame):
    empty_columns = df.columns[df.isnull().all()]
    column_stats = df.describe(include='all')
    result = {
        'no_of_unique_rows': int(df.shape[0]),
        'no_of_columns': int(df.shape[1]),
        'column_names': df.columns.tolist(),
        'empty_columns': 'None' if len(empty_columns) == 0 else empty_columns,
        'column_stats': [{x: {'count': int(column_stats.loc['count'][x]),
                              'unique': int(column_stats.loc['unique'][x]),
                              'most_common': column_stats.loc['top'][x],
                              'most_common_no_instances': int(column_stats.loc['freq'][x])}} for x in column_stats],
    }
    return result


def task_evidence_stats(df: pd.DataFrame):

    counts_teacher_project = df.groupby([df.uuid, df.project_title]).nunique()
    counts_teacher_project['task_or_project_evidence'] = counts_teacher_project['task_evidence'] + counts_teacher_project[
        'project_evidence']
    histogram_submitted_tasks = counts_teacher_project['tasks'].value_counts().sort_index().to_dict()
    histogram_submitted_evidences = counts_teacher_project['task_or_project_evidence'].value_counts().sort_index().to_dict()

    counts_project_task = df.groupby([df.project_title, df.tasks]).nunique()
    counts_project_task['task_or_project_evidence'] = counts_project_task['task_evidence'] \
                                                         + counts_project_task['project_evidence']
    counts_project_task['avg_evidences'] = counts_project_task['task_or_project_evidence'] / counts_project_task['uuid']

    result = {
        'histogram_of_tasks_submitted_per_teacher-project': histogram_submitted_tasks,
        'histogram_of_evidences_submitted_per_teacher-project': histogram_submitted_evidences,
        'avg_no_of_evidences_submitted_per_task': {str(i): row['avg_evidences'] for i, row in counts_project_task.iterrows()}
    }
    return result
