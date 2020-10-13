from itertools import tee


NEGATIVE_LABEL = -1
POSITIVE_LABEL = 1


def map_labels(labels, mapping):
    assert type(mapping) in [list, tuple]
    for label in labels:
        if label == mapping[0]:
            yield NEGATIVE_LABEL
        elif label == mapping[1]:
            yield POSITIVE_LABEL
        else:
            raise TypeError('Value not found in the mapping - map_labels() cannot map label classes.')


def get_or_default(options, key, default):
    if key in options:
        return options[key]
    else:
        return default


def pairwise(iterable):
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


# Print iterations progress
def print_progress_bar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='#'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end='\r')
    # Print New Line on Complete
    if iteration == total:
        print()
