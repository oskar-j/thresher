import pandas as pd


def get_sample_data():
    sample_data = pd.concat([pd.read_excel('./thresher/tests/negatives.xlsx', header=None, names=['pred', 'actual']),
                             pd.read_excel('./thresher/tests/negatives.xlsx', header=None, names=['pred', 'actual'])])
    return sample_data
