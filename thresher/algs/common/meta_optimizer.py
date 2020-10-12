import numpy as np


def get_mean_value_for_class_pd(label, label_column, data, data_column):
    return np.mean(data[data[label_column] == label][data_column])
