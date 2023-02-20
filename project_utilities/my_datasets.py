
from pandas import read_csv, read_pickle, DataFrame, concat

from dataclasses import dataclass

import preprocessing_functionality

'''@dataclass
class ITSupportDataset:
    """Class for storing the IT Support Ticket Descriptions, Impacts, Urgencies, and Overall Priority"""
    corpus = DataFrame
    raw_dataset = DataFrame

    def __init__(self, combined_title_description_requested: bool = False):
        self.__get_raw_dataset()
        self.__get_dataset(combined_title_description_requested)
        self.__add_overall_priority_column()

    def __get_raw_dataset(self):

        self.raw_dataset = read_csv('C:\\Users\\Benjamin\\PycharmProjects\\DISSERTATION_ARTEFACT\\project_utilities'
                                        '\\Datasets\\ITSupport_Tickets.csv')
        #ticket_data_high_prio = read_csv('C:\\Users\\Benjamin\\PycharmProjects\\DISSERTATION_ARTEFACT\\project_utilities'
                                         #'\\Datasets\\ITSupport_Tickets_High_Prio.csv')
        #self.raw_dataset = ticket_data_low_prio

    def __get_dataset(self, combined_title_description_requested: bool):
        impacts = self.raw_dataset['Impact'].tolist()
        urgencies = self.raw_dataset['Urgency'].tolist()
        texts = self.raw_dataset['Description'].tolist()
        if combined_title_description_requested:
            summaries = self.raw_dataset['Incident_Summary'].tolist()
            non_nulled_dataset = self.__remove_nulls_with_summaries(impacts, urgencies, texts, summaries)
        else:
            non_nulled_dataset = self.__remove_nulls(impacts, urgencies, texts)
        self.corpus = DataFrame(non_nulled_dataset)

    def __remove_nulls(self, impacts, urgencies, descriptions):
        dict_corpus = {'Descriptions': [], 'Impacts': [], 'Urgencies': []}
        for index in range(len(impacts)):
            if not (impacts[index] is np.nan
                    or urgencies[index] is np.nan
                    or descriptions[index] is np.nan):
                dict_corpus['Descriptions'].append(descriptions[index])
                dict_corpus['Impacts'].append(impacts[index])
                dict_corpus['Urgencies'].append(urgencies[index])

        return dict_corpus

    def __remove_nulls_with_summaries(self, impacts, urgencies, descriptions, summaries):
        dict_corpus = {'Descriptions': [], 'Impacts': [], 'Urgencies': []}

        for index in range(len(impacts)):
            if not (impacts[index] is np.nan
                    or urgencies[index] is np.nan
                    or descriptions[index] is np.nan):
                dict_corpus['Descriptions'].append(str(summaries[index]) + ' ' + str(descriptions[index]))
                dict_corpus['Impacts'].append(impacts[index])
                dict_corpus['Urgencies'].append(urgencies[index])

        return dict_corpus

    def __add_overall_priority_column(self):
        prio_to_num = {'Low': 0, 'Medium': 1, 'High': 2}
        num_to_pnum = ['P5', 'P4', 'P3', 'P2', 'P1']

        pnums = []
        for priorities in zip(self.corpus['Impacts'], self.corpus['Urgencies']):
            numbered_priority = sum([prio_to_num[priorities[0]], prio_to_num[priorities[1]]])
            pnums.append(num_to_pnum[numbered_priority])

        self.corpus['Priorities'] = pnums'''


@dataclass
class ITSupportDatasetWithBuilder:
    """Class for storing the IT Support Ticket Descriptions, Impacts, Urgencies, and Overall Priority
    Contains an associated Builder Class for flexible object creation."""
    corpus = DataFrame

    def __init__(self):
        self.__get_raw_dataset()
        self.__remove_nulls()

    def __get_raw_dataset(self):
        ticket_data_low_prio = read_csv('/\\project_utilities'
                                        '\\Datasets\\ITSupport_Tickets.csv')
        ticket_data_high_prio = read_csv('/\\project_utilities'
                                         '\\Datasets\\ITSupport_Tickets_High_Prio.csv')
        self.corpus = concat([ticket_data_low_prio, ticket_data_high_prio])

    def combine_summaries_with_descriptions(self):
        combined_columns = []
        for description, summary in zip(self.corpus['Description'].values, self.corpus['Incident_Summary'].values):
            combined_columns.append(str(summary) + ' ' + str(description))

        self.corpus['Description'] = combined_columns

    def __remove_nulls(self):
        self.corpus.replace('[None]', None, inplace=True)
        self.corpus.dropna(axis=0, subset=['Description', 'Impact', 'Urgency'], inplace=True, how='any')
        self.corpus.fillna('', axis=1, inplace=True)

    def add_overall_priority_column(self):
        prio_to_num = {'Low': 0, 'Medium': 1, 'High': 2}
        num_to_pnum = ['P5', 'P4', 'P3', 'P2', 'P1']

        pnums = []

        for impact, urgency, date in zip(
                self.corpus['Impact'].values, self.corpus['Urgency'].values, self.corpus['Added Date']):
            try:
                numbered_priority = sum([prio_to_num[impact], prio_to_num[urgency]])
                pnums.append(num_to_pnum[numbered_priority])
            except KeyError:
                print(date)

        self.corpus['Priority'] = pnums

    def pre_process_texts(self):
        self.corpus['Description'] = self.corpus['Description'].apply(preprocessing_functionality.clean_text)
        self.corpus['Description'] = self.corpus['Description'].str.split()
        self.corpus['Description'] = self.corpus['Description'].apply(preprocessing_functionality.stem_text)


class ITSupportDatasetBuilder(object):
    def __init__(self):
        self._dataset = ITSupportDatasetWithBuilder()

    def with_summaries_and_descriptions_combined(self):
        self._dataset.combine_summaries_with_descriptions()
        return self

    def with_overall_priority_column(self):
        self._dataset.add_overall_priority_column()
        return self

    def with_pre_processed_descriptions(self):
        self._dataset.pre_process_texts()
        return self

    def build(self):
        return self._dataset


'''
        #Previous method, more efficient, way more lines though
        impacts = self.raw_dataset['Impact'].tolist()
        urgencies = self.raw_dataset['Urgency'].tolist()
        descriptions = self.raw_dataset['Description'].tolist()
        summaries = self.raw_dataset['Incident_Summary'].tolist()'''

'''dict_corpus = {'Descriptions': [], 'Impacts': [], 'Urgencies': [], 'Summaries': []}
        start1, start2, end1, end2 = 0, 0, 0, 0
        start1 = time.perf_counter_ns()
        for description, impact, urgency, summary in zip(descriptions, impacts, urgencies, summaries):
            if not (impact is np.nan
                    or urgency is np.nan
                    or description is np.nan):
                dict_corpus['Descriptions'].append(description)
                dict_corpus['Impacts'].append(impact)
                dict_corpus['Urgencies'].append(urgency)
                dict_corpus['Summaries'].append(str(summary))
        end1 = time.perf_counter_ns()'''

# start2 = time.perf_counter_ns()
# self.corpus = self.raw_dataset
# end2 = time.perf_counter_ns()
# timing1, timing2 = end1 - start1, end2 - start2
# print(f"Iterative: {timing1}, Pandas: {timing2}, difference = {abs(timing1-timing2)}")
# return dict_corpus
if __name__ == '__main__':
    # obj = ITSupportDataset(combined_title_description_requested=False)
    '''times = []
    while True:
        for x in range(100):
            h1 = time.perf_counter_ns()
            dataset = ITSupportDatasetBuilder().with_summaries_and_descriptions_combined().with_overall_priority_column().build()
            times.append(time.perf_counter_ns() - h1)
        # dataset = ITSupportDatasetBuilder().with_overall_priority_column().build()
        print(numpy.mean(times))'''

    # dataset = ITSupportDatasetBuilder().with_summaries_and_descriptions_combined().with_overall_priority_column().build()
    '''ticket_data_low_prio = read_csv('C:\\Users\\Benjamin\\PycharmProjects\\DISSERTATION_ARTEFACT\\project_utilities'
                                    '\\Datasets\\ITSupport_Tickets.csv')
    ticket_data_high_prio = read_csv('C:\\Users\\Benjamin\\PycharmProjects\\DISSERTATION_ARTEFACT\\project_utilities'
                                     '\\Datasets\\ITSupport_Tickets_High_Prio.csv')
    corpus = concat([ticket_data_low_prio, ticket_data_high_prio])
    corpus.to_pickle('corpus.pickle')'''
    dataset = ITSupportDatasetBuilder().with_summaries_and_descriptions_combined().with_overall_priority_column().build()
    print(dataset.corpus.shape)
