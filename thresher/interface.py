#!/usr/bin/env python

import numpy as np
import pandas as pd
from collections.abc import Iterable

from thresher import algorithm
from thresher.oracle import run_oracle, run_computations
from thresher.exceptions import NOT_IMPLEMENTED_ERROR
from thresher.utils import map_labels


class ThresherBase(object):

    def _run_oracle(self, data_traits: dict):
        if self.options['algorithm'] == algorithm.DEFAULT:
            if self.options['verbose']:
                print("Running heuristics on choosing a proper algorithm")
            chosen_algorithm = run_oracle(data_traits)

        else:
            chosen_algorithm = self.options['algorithm']

        if self.options['verbose']:
            print(f"Chosen algorithm: {chosen_algorithm.full_name}")

        return chosen_algorithm

    def _compute(self, chosen_algorithm, scores, actual_classes):
        return run_computations(chosen_algorithm, scores, actual_classes,
                                self.options['verbose'],
                                self.options['progress_bar'],
                                self.options['allow_parallel'],
                                self.options['algorithm_params'])


class Thresher(ThresherBase):

    def __init__(self, **kwargs):
        """Creates a new Thresher object, an interface to the Thesher evaluator.

        The __init__ method creates the Thresher object.

        Note:
            No need to pass any extra arguments if you don't understand what you're doing

        Args:
            kwargs (:obj:`dict`, optional): Any hidden arguments, which you wish to pass.
        """

        super(ThresherBase, self).__init__()

        self.options = {
            'algorithm': 'auto',
            'allow_parallel': True,
            'verbose': False,
            'progress_bar': False,
            'algorithm_params': {}
        }

        self.options.update(kwargs)

        self.options['algorithm'] = algorithm.retrieve_by_alias(self.options['algorithm'])

    def get_current_algorithm(self):
        """Get current language."""
        with self.options['algorithm'] as current_algorithm:
            return {'name': current_algorithm.id, 'object': current_algorithm}

    def get_current_options(self):
        return self.options

    def set_algorithm(self, algorithm_name):
        try:
            self.options['algorithm'] = algorithm.retrieve_by_alias(algorithm_name)
        except StopIteration:
            print('Unknown algorithm, please use a name available in get_supported_algorithms()')
        return self

    @staticmethod
    def get_supported_algorithms(as_dict=False):
        """Get list of supported languages."""
        if as_dict:
            return {k: v.full_name for k, v in algorithm.available_algorithms.items()}
        else:
            return list(algorithm.available_algorithms.keys())

    def optimize_threshold(self, scores, actual_classes):
        if not isinstance(scores, Iterable):
            raise AttributeError(NOT_IMPLEMENTED_ERROR)
        if not isinstance(actual_classes, Iterable):
            raise AttributeError(NOT_IMPLEMENTED_ERROR)

        scores = list(scores)
        if ('labels' in self.options) and (isinstance(self.options['labels'], Iterable)):
            actual_classes = list(map_labels(actual_classes, self.options['labels']))
        else:
            actual_classes = list(actual_classes)

        data_traits = {'data_length': len(scores)}

        chosen_algorithm = self._run_oracle(data_traits)

        return self._compute(chosen_algorithm, scores, actual_classes)
