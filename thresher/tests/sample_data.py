import pandas as pd


def get_sample_data(path='./thresher/tests/'):
    sample_data = pd.concat([pd.read_excel(f'{path}negatives.xlsx', header=None, names=['pred', 'actual']),
                             pd.read_excel(f'{path}negatives.xlsx', header=None, names=['pred', 'actual'])])
    return sample_data
