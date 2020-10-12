import numpy as np


def calculate_range_mean(scores, actual_classes, label):
    return np.mean([_[0] for _ in zip(scores, actual_classes) if _[1] == label])


def get_mean_value_for_class_pd(label, label_column, data, data_column):
    return np.mean(data[data[label_column] == label][data_column])
