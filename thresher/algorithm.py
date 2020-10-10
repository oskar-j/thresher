from collections import namedtuple

Algorithm = namedtuple('Algorithm', ['id', 'synonyms', 'data_vol_thresh'])

available_algorithms = {'Linear search': Algorithm(id='ls', synonyms=['linear', 'linear_search'],
                                                   data_vol_thresh=500),
                        'Stochastic gradient descent': Algorithm(id='sgd', synonyms=['curve_fitting'],
                                                                 data_vol_thresh=10*1000),
                        'Genetic algorithm': Algorithm(id='gen', synonyms=['genetic', 'sim'],
                                                       data_vol_thresh=None),
                       }
