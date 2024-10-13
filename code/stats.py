import os

import matplotlib.pyplot as plt
import pandas as pd


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
                              # 'most_common_no_instances': int(column_stats.loc['freq'][x])
                              }} for x in column_stats
                         ],
    }
    return result


def task_evidence_stats(df: pd.DataFrame, plot_title_string='Untitled', output_dir='.',
                        create_plots=False, show_plots=False, save_plots=False):

    block_projects, district_projects, state_projects = get_projects_by_geography(df)

    counts_teacher_project, histogram_submitted_evidences, histogram_submitted_tasks = get_project_stats(df)

    counts_project_task = df.groupby([df.project_title, df.tasks]).nunique()
    counts_project_task['task_or_project_evidence'] = \
        counts_project_task['task_evidence'] + counts_project_task['project_evidence']
    task_counts = df.groupby([df.project_title, df.tasks]).agg({'uuid': 'count'})

    # Remove the custom tasks added by users and only retain the original task description that is used by most users
    threshold = sum(task_counts.uuid) * 0.025
    default_tasks = task_counts[task_counts['uuid'] > threshold]

    percent_teacher_tasks_with_evidence = {str(t): None for t in default_tasks.index}
    for t in default_tasks.index:
        df_filtered = df.loc[df['project_title'] == t[0]]
        df_filtered = df_filtered.loc[df['tasks'] == t[1]]
        unique = df_filtered['uuid'].nunique()
        df_filtered_with_evidence = df_filtered.loc[(pd.notna(df_filtered['task_evidence']))
                                                    | (pd.notna(df_filtered['project_evidence']))]
        unique_with_evidence = df_filtered_with_evidence['uuid'].nunique()
        task_evidence_percent = round(100 * unique_with_evidence / unique, 0)
        percent_teacher_tasks_with_evidence[str(t)] = task_evidence_percent

    result = {
        'activities': list(df['project_title'].dropna().unique()),
        'no_of_default_tasks_for_activity': len(default_tasks.index),
        'no_of_users': df.uuid.nunique(),
        'no_of_teachers': df[df.user_type == 'teacher']['uuid'].nunique(),
        'no_of_administrators': df[df.user_type == 'administrator']['uuid'].nunique(),
        'no_of_projects': counts_teacher_project.shape[0],
        'no_of_projects_with_evidence': len(counts_teacher_project[counts_teacher_project['task_or_project_evidence'] > 0]),
        'total_no_of_tasks_across_all_projects': df.shape[0],
        'no_of_tasks_with_evidence': len(df[pd.notna(df['task_evidence']) | pd.notna(df['project_evidence'])]),
        'no_of_states': df.declared_state.nunique(),
        'no_of_districts': df['state-district'].nunique(),
        'no_of_blocks': df['state-district-block'].nunique(),
        'no_of_projects_by_state': state_projects,
        'no_of_projects_by_district': district_projects,
        'no_of_projects_by_block': block_projects,
    }

    if create_plots:
        # Plot project engagement
        plot_project_engagement(task_data=histogram_submitted_evidences,
                                no_of_default_tasks=result['no_of_default_tasks'],
                                projects=result['projects'],
                                states=result['states'],
                                districts=result['districts'],
                                blocks=result['blocks'],
                                title_string=plot_title_string,
                                output_dir=output_dir,
                                show=show_plots,
                                save=save_plots)
        # Plot task-wise engagement
        plot_task_wise_engagement(task_data=percent_teacher_tasks_with_evidence,
                                  projects=result['projects'],
                                  states=result['states'],
                                  districts=result['districts'],
                                  blocks=result['blocks'],
                                  title_string=plot_title_string,
                                  output_dir=output_dir,
                                  show=show_plots,
                                  save=save_plots)

    return result


def plot_district_participation():
    pass


def plot_block_participation():
    pass


def plot_project_engagement(task_data, no_of_default_tasks, projects, states, districts, blocks, title_string, output_dir,
                            show=True, save=False):
    if len(projects) > 1:
        projects = len(projects)
    if len(states) > 1: states = len(states)
    if len(districts) > 1: districts = len(districts)
    if len(blocks) > 1: blocks = len(blocks)
    keys = task_data.keys()
    values = task_data.values()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
    fig.suptitle('User Engagement Distribution\nProject: {}\nState(s): {}  |  District(s): {}  |  Block(s): {}'.format(projects, states, districts, blocks))
    low_engagement_threshold = round(0.4 * no_of_default_tasks, 0)
    med_engagement_threshold = round(0.8 * no_of_default_tasks, 0)
    colors = ['red' if x <= low_engagement_threshold else ('orange' if x <= med_engagement_threshold else 'green') for x
              in keys]
    ax1.bar(keys, values, color=colors)
    ax1.axvline(x=no_of_default_tasks, color='k', linestyle='--', label='# of default tasks')
    ax1.set_title('Frequency of evidences submitted per project')
    ax1.set_xlabel('No. of evidences submitted')
    ax1.set_ylabel('No. of projects')
    low_engagement_percent = round(
        100 * sum([task_data[x] for x in keys if x <= low_engagement_threshold]) / sum(values), 0)
    med_engagement_percent = round(
        100 * sum([task_data[x] for x in keys if low_engagement_threshold < x <= med_engagement_threshold]) / sum(
            values), 0)
    high_engagement_percent = round(
        100 * sum([task_data[x] for x in keys if x > med_engagement_threshold]) / sum(values), 0)
    ax2.pie([low_engagement_percent, med_engagement_percent, high_engagement_percent],
            colors=['red', 'orange', 'green'],
            labels=['Low Engagement', 'Medium Engagement', 'High Engagement'], autopct='%1.0f%%', shadow=True,
            startangle=140)
    ax2.axis('equal')
    ax2.set_title('Engagement Percentages')
    fig.subplots_adjust(top=0.836, bottom=0.1, left=0.125, right=0.9, wspace=0.4, hspace=0.2)
    if save:
        plt.savefig(os.path.join(output_dir, '{}_{}.png'.format(title_string, 'ProjectEngagement')), format='png', dpi=300)
    if show:
        plt.show()


def plot_task_wise_engagement(task_data, projects, states, districts, blocks, title_string, output_dir,
                              show=True, save=False):
    if len(projects) > 1: projects = len(projects)
    if len(states) > 1: states = len(states)
    if len(districts) > 1: districts = len(districts)
    if len(blocks) > 1: blocks = len(blocks)
    sorted_items = sorted(task_data.items(), key=lambda item: item[0], reverse=True)  # Sort by keys
    sorted_keys, sorted_values = zip(*sorted_items)
    fig, ax = plt.subplots(figsize=(12, 8))
    fig.suptitle('Task-wise Engagement\nProject(s): {}\nState(s): {}  |  District(s): {}  |  Block(s): {}'.format(projects, states, districts, blocks))
    colors = ['red' if task_data[x] <= 30 else ('orange' if task_data[x] <= 60 else 'green') for x in sorted_keys]
    ax.barh(sorted_keys, sorted_values, color=colors)
    ax.set_title('Evidence submission by Task')
    ax.set_xlabel('Percent of Teachers who submitted at least one Evidence')
    ax.set_ylabel('Task')
    fig.subplots_adjust(top=0.836, bottom=0.1, left=0.675, right=0.95, wspace=0.2, hspace=0.2)
    if save:
        plt.savefig(os.path.join(output_dir, '{}_{}.png'.format(title_string, 'TaskEngagement')),
                    format='png', dpi=300)
    if show:
        plt.show()


def plot_district_engagement():
    pass


def plot_block_engagement():
    pass


def get_project_stats(df):
    counts_teacher_project = df.groupby([df.uuid, df.project_title]).nunique()
    counts_teacher_project['task_or_project_evidence'] = \
        counts_teacher_project['task_evidence'] + counts_teacher_project['project_evidence']
    histogram_submitted_tasks = counts_teacher_project['tasks'].value_counts().sort_index().to_dict()
    histogram_submitted_evidences = counts_teacher_project[
        'task_or_project_evidence'].value_counts().sort_index().to_dict()
    return counts_teacher_project, histogram_submitted_evidences, histogram_submitted_tasks


def get_projects_by_geography(df):
    df['teacher-project'] = df['uuid'] + df['project_title']
    state_projects = df.groupby(df.declared_state)['teacher-project'].nunique().to_dict()
    df['state-district'] = df['declared_state'] + df['district']
    district_projects = df.groupby([df.declared_state, df.district])['teacher-project'].nunique().to_dict()
    district_projects = {str(k): v for k, v in district_projects.items()}
    df['state-district-block'] = df['declared_state'] + df['district'] + df['block']
    block_projects = df.groupby([df.declared_state, df.district, df.block])['teacher-project'].nunique().to_dict()
    block_projects = {str(k): v for k, v in block_projects.items()}
    return block_projects, district_projects, state_projects
