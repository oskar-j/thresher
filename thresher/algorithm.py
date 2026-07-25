from collections import namedtuple

from thresher.exceptions import UNKNOWN_ALGORITHM_NAME

Algorithm = namedtuple('Algorithm', ['id', 'full_name', 'synonyms', 'data_vol_thresh'])

available_algorithms = {'auto': Algorithm(id='auto', synonyms=['default', 'default_heuristics'],
                                          data_vol_thresh=None, full_name='Default heuristics'),
                        'ls': Algorithm(id='ls', synonyms=['linear', 'linear_search'],
                                        data_vol_thresh=1000, full_name='Linear search'),
                        'sgd': Algorithm(id='sgd', synonyms=['curve_fitting', ],
                                         data_vol_thresh=None, full_name='Stochastic gradient descent'),
                        'gen': Algorithm(id='gen', synonyms=['genetic', 'sim'],
                                         data_vol_thresh=None, full_name='Genetic algorithm'),
                        'grid': Algorithm(id='grid', synonyms=['grid-search', 'gs'],
                                          data_vol_thresh=50*1000, full_name='Grid search'),
                        'sgrid': Algorithm(id='sgrid', synonyms=['random-grid-search', 'rn-grid', 's-grid'],
                                           data_vol_thresh=None, full_name='Stochastic grid search'),
                        }

DEFAULT = available_algorithms['auto']


def retrieve_by_alias(name: str):
    name = name.lower()
    try:
        return available_algorithms[name]
    except KeyError:
        # try to match by the 'alternate name'
        try:
            return next(_ for _ in available_algorithms.values() if name in _.synonyms)
        except StopIteration:
            # 'next' on an exhausted generator raises StopIteration, which says nothing
            # about what went wrong and reads as a bug rather than a bad argument.
            raise ValueError(UNKNOWN_ALGORITHM_NAME.format(
                name=name, available=', '.join(sorted(available_algorithms)))) from None
