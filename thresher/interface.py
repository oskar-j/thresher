#!/usr/bin/env python

import numpy as np
import pandas as pd
from collections.abc import Iterable

from thresher import algorithm
from thresher.oracle import run_oracle, run_computations
from thresher.exceptions import NOT_IMPLEMENTED_ERROR


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
                                self.options['verbose'], self.options['progress_bar'])


class Thresher(ThresherBase):

    def __init__(self, **kwargs):
        """Creates a new FastCast object, an interface to Wikipedia categories.

        The __init__ method creates the FastCat object, which acts as the main and only
        interface in communicating with the Redis server (where the Wikipedia categories are
        first loaded (stored) and then retrieved on demand). Default settings mean that you're connecting
        to a Redis instance on the localhost and 6379 port, and your current language of categories is English.
        It's possible to pass your own Redis client object into the 'db' arg, or,
        alternatively, custom args to the Redis client __init__ method.

        Note:
            No need to pass any extra arguments if you don't understand what you're doing

        Args:
            db (:obj:`Redis`, optional): Custom Redis client.
            language (:obj:`str`, optional): Choose the default language.
            kwargs (:obj:`dict`, optional): Any arguments, which you wish to pass to the Redis client.

        """

        super(ThresherBase, self).__init__()

        self.options = {
            'algorithm': 'auto',
            'verbose': False,
            'progress_bar': False
        }

        self.options.update(kwargs)

        self.options['algorithm'] = algorithm.retrieve_by_alias(self.options['algorithm'])

    def get_current_algorithm(self):
        """Get current language."""
        with self.options['algorithm'] as current_algorithm:
            return {'name': current_algorithm.id, 'object': current_algorithm}

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

        data_traits = {'data_length': len(scores)}

        chosen_algorithm = self._run_oracle(data_traits)

        return self._compute(chosen_algorithm, scores, actual_classes)
