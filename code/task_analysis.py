import json
import os
from collections import Counter
import yaml

import matplotlib.pyplot as plt
import openai
import pandas as pd


class TaskAnalysis:

    def __init__(self, dataset_name: str, df: pd.DataFrame):
        self.dataset_name = dataset_name
        self.df = df
        self.activities = list(df['project_title'].dropna().unique())
        self.default_tasks = self._get_default_tasks()
        self.no_of_default_tasks = len(self.default_tasks)
        self.no_of_users = df.uuid.nunique()
        self.no_of_teachers = df[df.user_type == 'teacher']['uuid'].nunique()
        self.no_of_administrators = df[df.user_type == 'administrator']['uuid'].nunique()
        self.no_of_projects_by_block, self.no_of_projects_by_district, self.no_of_projects_by_state = \
            self._get_projects_by_geography()
        self.counts_teacher_project, self.histogram_submitted_evidences, self.histogram_submitted_tasks = \
            self._get_project_stats()
        self.no_of_states = df.declared_state.nunique()
        self.no_of_districts = df['state-district'].nunique()
        self.no_of_blocks = df['state-district-block'].nunique()
        self.no_of_projects = self.counts_teacher_project.shape[0]
        self.no_of_projects_with_evidence = len(
            self.counts_teacher_project[self.counts_teacher_project['task_or_project_evidence'] > 0])
        self.total_no_of_tasks_across_all_projects = df.shape[0]
        self.no_of_tasks_with_evidence = len(df[pd.notna(df['task_evidence']) | pd.notna(df['project_evidence'])])
        tasks_with_remarks = df[pd.notna(df.task_remarks) | pd.notna(df.project_remarks)]
        tasks_with_remarks['combined_remarks'] = tasks_with_remarks[['task_remarks', 'project_remarks']].agg(
            lambda x: '; '.join(x.dropna()), axis=1)
        self.no_of_tasks_with_remarks = tasks_with_remarks.shape[0]
        self.percent_tasks_with_remarks = round(100 * self.no_of_tasks_with_remarks / df.shape[0])
        self.no_of_teachers_with_remarks = tasks_with_remarks.uuid.nunique()
        self.percent_teachers_with_remarks = round(100 * self.no_of_tasks_with_remarks / df.uuid.nunique())
        self.remark_scores, self.no_of_very_positive_remarks, self.no_of_moderately_positive_remarks, \
        self.no_of_neutral_remarks, \
        self.no_of_moderately_negative_remarks, self.no_of_very_negative_remarks = \
            TaskAnalysis._run_sentiment_analysis(tasks_with_remarks)

    def __repr__(self):
        return {
            'activities': self.activities,
            'no_of_default_tasks': self.no_of_default_tasks,
            'no_of_users': self.no_of_users,
            'no_of_teachers': self.no_of_teachers,
            'no_of_administrators': self.no_of_administrators,
            'no_of_projects': self.no_of_projects,
            'no_of_projects_with_evidence': self.no_of_projects_with_evidence,
            'total_no_of_tasks_across_all_projects': self.total_no_of_tasks_across_all_projects,
            'no_of_tasks_with_evidence': self.no_of_tasks_with_evidence,
            'no_of_states': self.no_of_states,
            'no_of_districts': self.no_of_districts,
            'no_of_blocks': self.no_of_blocks,
            'no_of_projects_by_state': self.no_of_projects_by_state.reset_index(),
            'no_of_projects_by_district': self.no_of_projects_by_district.reset_index(),
            'no_of_projects_by_block': self.no_of_projects_by_block.reset_index(),
            'total_no_of_remarks': self.remark_scores.shape[0],
            'no_of_very_positive_remarks': self.no_of_very_positive_remarks,
            'no_of_moderately_positive_remarks': self.no_of_moderately_positive_remarks,
            'no_of_neutral_remarks': self.no_of_neutral_remarks,
            'no_of_moderately_negative_remarks': self.no_of_moderately_negative_remarks,
            'no_of_very_negative_remarks': self.no_of_very_negative_remarks,
            'remarks_list_with_sentiment_scores': self.remark_scores.reset_index()
        }

    def visualize(self, dataset_name, output_dir, show_plots=True, save_plots=False):

        self._plot_project_engagement(
            task_data=self.histogram_submitted_evidences,
            title_string=dataset_name,
            output_dir=output_dir,
            show=show_plots,
            save=save_plots)

        self._plot_task_wise_engagement(
            task_data=self._get_task_wise_evidence(),
            title_string=dataset_name,
            output_dir=output_dir,
            show=show_plots,
            save=save_plots)

        self._plot_remark_sentiment_scores(
            remarks_data=list(self.remark_scores['sentiment_score']),
            title_string=dataset_name,
            output_dir=output_dir,
            show=show_plots,
            save=save_plots
        )

    def _get_project_stats(self):
        counts_teacher_project = self.df.groupby([self.df.uuid, self.df.project_title]).nunique()
        counts_teacher_project['task_or_project_evidence'] = \
            counts_teacher_project['task_evidence'] + counts_teacher_project['project_evidence']
        histogram_submitted_tasks = counts_teacher_project['tasks'].value_counts().sort_index().to_dict()
        histogram_submitted_evidences = counts_teacher_project[
            'task_or_project_evidence'].value_counts().sort_index().to_dict()
        return counts_teacher_project, histogram_submitted_evidences, histogram_submitted_tasks

    def _get_projects_by_geography(self):
        self.df['teacher-project'] = self.df['uuid'] + self.df['project_title']
        state_projects = self.df.groupby(self.df.declared_state)['teacher-project'].nunique().to_frame()
        self.df['state-district'] = self.df['declared_state'] + self.df['district']
        district_projects = self.df.groupby([self.df.declared_state, self.df.district])[
            'teacher-project'].nunique().to_frame()
        self.df['state-district-block'] = self.df['declared_state'] + self.df['district'] + self.df['block']
        block_projects = self.df.groupby([self.df.declared_state, self.df.district, self.df.block])[
            'teacher-project'].nunique().to_frame()
        return block_projects, district_projects, state_projects

    def _get_default_tasks(self):
        task_counts = self.df.groupby([self.df.project_title, self.df.tasks]).agg({'uuid': 'count'})
        removal_threshold = sum(task_counts.uuid) * 0.025
        default_tasks_counts = task_counts[task_counts['uuid'] > removal_threshold]
        default_tasks = default_tasks_counts.index
        return default_tasks

    def _get_task_wise_evidence(self):
        percent_teacher_tasks_with_evidence = {str(t): None for t in self.default_tasks}
        for t in self.default_tasks:
            df_filtered = self.df.loc[self.df['project_title'] == t[0]]
            df_filtered = df_filtered.loc[self.df['tasks'] == t[1]]
            unique = df_filtered['uuid'].nunique()
            df_filtered_with_evidence = df_filtered.loc[(pd.notna(df_filtered['task_evidence']))
                                                        | (pd.notna(df_filtered['project_evidence']))]
            unique_with_evidence = df_filtered_with_evidence['uuid'].nunique()
            task_evidence_percent = round(100 * unique_with_evidence / unique, 0)
            percent_teacher_tasks_with_evidence[str(t)] = task_evidence_percent
        return percent_teacher_tasks_with_evidence

    @staticmethod
    def _run_sentiment_analysis(df_tasks_with_remarks):

        def get_sentiment(df, client):
            try:
                text = list(df['combined_remarks'])
                model = 'gpt-3.5-turbo'
                temperature = 0.7
                max_tokens = 4096
                response = client.chat.completions.create(
                    model=model,  # or use 'gpt-4' if you have access
                    messages=[
                        {'role': 'user',
                         'content': 'You are an expert sentiment analyzer. '
                                    'I will give you a list of remarks added by teachers who have run an activity '
                                    'for their students and are talking about their experience, '
                                    'how easy/difficult the activity was, what exactly they or their students did, '
                                    'or how the students responded to the activity. '
                                    'I want you to tell me the sentiment of each remark in a one word response '
                                    'of either "very positive", "positive", "neutral", "negative", or "very negative". '
                                    'Return the response as a JSON list  of objects with a remark and sentiment field like this: '
                                    '{{\"result\": [{{\"remark\": (remark), \"sentiment\": (sentiment)}}]}}. '
                                    'If you cannot figure out the sentiment for a remark, just mark that one as neutral. '
                                    'But make sure that there are exactly the same number of items in the '
                                    'sentiments list as in the input list of remarks.'
                                    'Here is the list of remarks to be analyzed: {}'.format(text)}
                    ],
                    temperature=temperature,  # Adjust the creativity of responses (0-1)
                    max_tokens=max_tokens,  # Limit the length of the response
                    n=1,  # Number of responses to generate
                    stop=None  # You can set a stopping sequence if desired
                )
                print('\nOpenAI Completion:\nModel: {}, Temperature: {}, Max tokens: {}\nUsage: {}\n'.format(response.model,
                                                                                                         temperature,
                                                                                                         max_tokens,
                                                                                                         response.usage))

                return response.choices[0].message.content

            except Exception as e:
                print(f"Error: {e}")
                return None

        tasks_with_remarks = df_tasks_with_remarks[
            pd.notna(df_tasks_with_remarks.task_remarks) | pd.notna(df_tasks_with_remarks.project_remarks)]
        tasks_with_remarks['sentiment'] = None

        try:
            with open('./secret.yaml', 'r') as sf:
                secrets = yaml.load(sf, Loader=yaml.SafeLoader)
                openai_api_key = secrets['openai_api_key']
                client = openai.OpenAI(
                    api_key=openai_api_key,
                )
        except Exception as e:
            print('ERROR: Could not create OpenAPI client. Check whether there is a secret.yaml file with all the '
                  'required properties. Also check, if there is a valid OpenAI API key.')
            raise e

        def get_sentiment_in_df_chunks(df, chunk_size, client):
            results = dict()
            for start in range(0, len(df), chunk_size):
                end = start + chunk_size
                chunk = df.iloc[start:end]
                response = get_sentiment(chunk, client)
                try:
                    sentiments_result = json.loads(response)
                    sentiments_dict = {i['remark']: i['sentiment'] for i in sentiments_result['result']}
                    results.update(sentiments_dict)
                except Exception as e:
                    print(e)
            return results

        chunk_size = 50  # adjust based on what will fit within LL
        results = get_sentiment_in_df_chunks(tasks_with_remarks, chunk_size, client)
        tasks_with_remarks['sentiment'] = tasks_with_remarks['combined_remarks'].map(results).fillna('neutral')
        tasks_with_remarks['sentiment_score'] = tasks_with_remarks['sentiment'].replace({'very positive': 1,
                                                                                         'positive': 0.5,
                                                                                         'neutral': 0,
                                                                                         'negative': -0.5,
                                                                                         'very negative': -1})

        remark_scores = tasks_with_remarks[['combined_remarks', 'sentiment_score']].sort_values(by='sentiment_score')
        remark_scores.set_index('combined_remarks', inplace=True)
        no_of_very_positive_remarks = remark_scores[remark_scores.sentiment_score > 0.5].shape[0]
        no_of_moderately_positive_remarks = remark_scores[(0 < remark_scores.sentiment_score)
                                                          & (remark_scores.sentiment_score <= 0.5)].shape[0]
        no_of_neutral_remarks = remark_scores[(-0.1 <= remark_scores.sentiment_score)
                                              & (remark_scores.sentiment_score <= 0.1)].shape[0]
        no_of_moderately_negative_remarks = \
            remark_scores[(remark_scores.sentiment_score < -0.1) & (remark_scores.sentiment_score >= -0.5)].shape[0]
        no_of_very_negative_remarks = remark_scores[remark_scores.sentiment_score < -0.5].shape[0]

        return remark_scores, \
               no_of_very_positive_remarks, \
               no_of_moderately_positive_remarks, \
               no_of_neutral_remarks, \
               no_of_moderately_negative_remarks, \
               no_of_very_negative_remarks

    def _plot_project_engagement(self, task_data, title_string, output_dir, show=True, save=False):
        keys = task_data.keys()
        values = task_data.values()
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 9))
        fig.suptitle(
            'User Engagement Distribution\nProject: {}\nState(s): {}  |  District(s): {}  |  Block(s): {}'.format(
                self.no_of_projects, self.no_of_states, self.no_of_districts, self.no_of_blocks))
        low_engagement_threshold = round(0.4 * self.no_of_default_tasks, 0)
        med_engagement_threshold = round(0.8 * self.no_of_default_tasks, 0)
        colors = ['red' if x <= low_engagement_threshold else ('orange' if x <= med_engagement_threshold else 'green')
                  for x
                  in keys]
        ax1.bar(keys, values, color=colors)
        ax1.axvline(x=self.no_of_default_tasks, color='k', linestyle='--', label='# of default tasks')
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
            plt.savefig(os.path.join(output_dir, '{}__{}.png'.format(title_string, 'ProjectEngagement')), format='png',
                        dpi=300)
        if show:
            plt.show()

    def _plot_task_wise_engagement(self, task_data, title_string, output_dir, show=True, save=False):
        sorted_items = sorted(task_data.items(), key=lambda item: item[0], reverse=True)  # Sort by keys
        sorted_keys, sorted_values = zip(*sorted_items)
        fig, ax1 = plt.subplots(figsize=(12, 8))
        fig.suptitle(
            'Task-wise Engagement\nProject(s): {}\nState(s): {}  |  District(s): {}  |  Block(s): '
            '{}'.format(self.no_of_projects, self.no_of_states, self.no_of_districts, self.no_of_blocks))
        colors = ['red' if task_data[x] <= 30 else ('orange' if task_data[x] <= 60 else 'green') for x in sorted_keys]
        ax1.barh(sorted_keys, sorted_values, color=colors)
        ax1.set_title('Evidence submission by Task')
        ax1.set_xlabel('Percent of Teachers who submitted at least one Evidence')
        ax1.set_ylabel('Task')

        fig.subplots_adjust(top=0.836, bottom=0.1, left=0.675, right=0.95, wspace=0.2, hspace=0.2)
        if save:
            plt.savefig(os.path.join(output_dir, '{}__{}.png'.format(title_string, 'TaskEngagement')),
                        format='png', dpi=300)
        if show:
            plt.show()

    def _plot_remark_sentiment_scores(self, remarks_data, title_string, output_dir, show=True, save=False):
        score_counts = Counter(remarks_data)
        score_list = [score_counts.get(score, 0) for score in [1, 0.5, 0, -0.5, -1]]
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
        fig.suptitle(
            'User Sentiment\nProject(s): {}\nState(s): {}  |  District(s): {}  |  Block(s): '
            '{}'.format(self.no_of_projects, self.no_of_states, self.no_of_districts, self.no_of_blocks))
        ax1.barh(width=score_list, y=['Very Positive', 'Positive', 'Neutral', 'Negative', 'Very Negative'],
                 color=['g', 'lightgreen', 'lightblue', 'orange', 'red'])
        ax1.set_title('User Sentiment Scores')
        ax1.set_xlabel('Remarks')
        ax1.set_ylabel('Sentiment Score')

        ax2.pie(score_list,
                colors=['g', 'lightgreen', 'lightblue', 'orange', 'red'],
                shadow=True,
                autopct='%0.1f %%',
                startangle=140)
        labels = ['Very Positive', 'Positive', 'Neutral', 'Negative', 'Very Negative']
        ax2.legend(labels, loc="best", bbox_to_anchor=(1, 0.5))
        ax2.axis('equal')
        ax2.set_title('User Sentiment Score Distribution')

        fig.subplots_adjust(top=0.836, bottom=0.1, left=0.125, right=0.8, wspace=0.6, hspace=0.2)
        if save:
            plt.savefig(os.path.join(output_dir, '{}__{}.png'.format(title_string, 'UserSentiment')),
                        format='png', dpi=300)
        if show:
            plt.show()

    # TODO
    def _plot_district_participation(self):
        pass

    def _plot_block_participation(self):
        pass

    def _plot_district_engagement(self):
        pass

    def _plot_block_engagement(self):
        pass
