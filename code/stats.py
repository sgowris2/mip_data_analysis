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

    df['teacher-project'] = df['uuid'] + df['project_title']
    state_projects = df.groupby(df.declared_state)['teacher-project'].nunique().to_dict()

    df['state-district'] = df['declared_state'] + df['district']
    district_projects = df.groupby([df.declared_state, df.district])['teacher-project'].nunique().to_dict()
    district_projects = {str(k): v for k, v in district_projects.items()}

    df['state-district-block'] = df['declared_state'] + df['district'] + df['block']
    block_projects = df.groupby([df.declared_state, df.district, df.block])['teacher-project'].nunique().to_dict()
    block_projects = {str(k): v for k, v in block_projects.items()}

    counts_teacher_project = df.groupby([df.uuid, df.project_title]).nunique()
    counts_teacher_project['task_or_project_evidence'] = \
        counts_teacher_project['task_evidence'] + counts_teacher_project['project_evidence']
    histogram_submitted_tasks = counts_teacher_project['tasks'].value_counts().sort_index().to_dict()
    histogram_submitted_evidences = counts_teacher_project['task_or_project_evidence'].value_counts().sort_index().to_dict()

    counts_project_task = df.groupby([df.project_title, df.tasks]).nunique()
    counts_project_task['task_or_project_evidence'] = \
        counts_project_task['task_evidence'] + counts_project_task['project_evidence']

    task_counts = df.groupby([df.project_title, df.tasks]).agg({'uuid': 'count'})
    threshold = sum(task_counts.uuid) * 0.025
    major_tasks = task_counts[task_counts['uuid'] > threshold]
    counts_project_task = counts_project_task.loc[major_tasks.index]

    counts_project_task['avg_evidences'] = \
        counts_project_task['task_or_project_evidence'] / counts_project_task['uuid']

    result = {
        'no_of_states': df.declared_state.nunique(),
        'no_of_districts': df.district.nunique(),
        'no_of_users': df.uuid.nunique(),
        'no_of_teachers': df[df.user_type == 'teacher']['uuid'].nunique(),
        'no_of_administrators': df[df.user_type == 'administrator']['uuid'].nunique(),
        'no_of_projects': counts_teacher_project.shape[0],
        'no_of_projects_by_state': state_projects,
        'no_of_projects_by_district': district_projects,
        'no_of_projects_by_block': block_projects,
        'histogram_of_tasks_submitted_per_teacher-project': histogram_submitted_tasks,
        'histogram_of_evidences_submitted_per_teacher-project': histogram_submitted_evidences,
        'avg_no_of_evidences_submitted_per_task': {str(i): round(row['avg_evidences'], 2) for i, row in counts_project_task.iterrows()}
    }

    hist_dict = result['histogram_of_tasks_submitted_per_teacher-project']
    fig, ax = plt.subplots()
    ax.bar(hist_dict.keys(), hist_dict.values())
    ax.set_xticks(list(hist_dict.keys()))
    ax.set_title('Frequency of tasks submitted per project')
    ax.set_xlabel('No. of tasks submitted')
    ax.set_ylabel('No. of projects')

    hist_dict = result['histogram_of_evidences_submitted_per_teacher-project']
    fig2, ax2 = plt.subplots()
    ax2.bar(hist_dict.keys(), hist_dict.values())
    ax2.set_title('Frequency of evidences submitted per project')
    ax2.set_xlabel('No. of evidences submitted')
    ax2.set_ylabel('No. of projects')

    hist_dict = result['avg_no_of_evidences_submitted_per_task']
    hist_dict = {k: v for k, v in hist_dict.items() if '\' Quadrilaterals\'' in k and '. ' in k}
    sorted_items = sorted(hist_dict.items(), key=lambda item: item[0])  # Sort by values
    sorted_keys, sorted_values = zip(*sorted_items)
    fig3, ax3 = plt.subplots()
    ax3.barh(sorted_keys, sorted_values)
    ax3.set_title('Average # evidences per task')
    ax3.set_xlabel('Average # evidences')
    ax3.set_ylabel('Task')
    plt.show()

    return result
