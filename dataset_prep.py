from pandas import read_csv, DataFrame, concat
import numpy as np


def getDataset():
    ticket_data = getRawDataset()
    impacts = ticket_data['Impact'].tolist()
    urgencies = ticket_data['Urgency'].tolist()
    texts = ticket_data['Description'].tolist()

    dict_corpus = {'Descriptions': [], 'Impacts': [], 'Urgencies': []}

    for index in range(len(impacts)):
        if not (impacts[index] is np.nan
                or urgencies[index] is np.nan
                or texts[index] is np.nan):
            dict_corpus['Descriptions'].append(texts[index])
            dict_corpus['Impacts'].append(impacts[index])
            dict_corpus['Urgencies'].append(urgencies[index])

    data_frame_corpus = DataFrame(dict_corpus)
    return data_frame_corpus


def getRawDataset():
    ticket_data_low_prio = read_csv('project_utilities/Datasets/ITSupport_Tickets.csv')
    ticket_data_high_prio = read_csv('custom_models/ITSupport_Tickets_High_Prio.csv')
    ticket_data_whole = concat([ticket_data_low_prio, ticket_data_high_prio])
    return ticket_data_whole


def convertToPriorities(dataset: DataFrame or dict) -> DataFrame:
    prio_to_num = {'Low': 0, 'Medium': 1, 'High': 2}
    num_to_pnum = ['P5', 'P4', 'P3', 'P2', 'P1']

    pnums = []
    for priorities in zip(dataset['Impacts'], dataset['Urgencies']):
        numbered_priority = sum([prio_to_num[priorities[0]], prio_to_num[priorities[1]]])
        pnums.append(num_to_pnum[numbered_priority])

    dataset['Priorities'] = pnums
    return dataset


if __name__ == '__main__':
    hi = getDataset()
    print(convertToPriorities(hi))
