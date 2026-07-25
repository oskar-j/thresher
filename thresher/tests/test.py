import random
import subprocess
import sys
import unittest
import thresher
from thresher import algorithm
from thresher.oracle import run_oracle
from thresher.tests.sample_data import get_sample_data


class ThresherMediumTest(unittest.TestCase):

    def setUp(self):
        # Preparing data for unit test
        self.t = thresher.Thresher(verbose=False, progress_bar=False)
        self.alt_t = thresher.Thresher(algorithm='linear', verbose=False, progress_bar=False)
        self.alt_t2 = thresher.Thresher(algorithm='sim', verbose=False, progress_bar=False)
        self.alt_t3 = thresher.Thresher(algorithm='grid')
        self.alt_t4 = thresher.Thresher(algorithm='sgrid',
                                        algorithm_params={'no_of_decimal_places': 2,
                                                          'stoch_ratio': 0.10})
        self.alt_t5 = thresher.Thresher(algorithm='sgrid',
                                        algorithm_params={'no_of_decimal_places': 3,
                                                          'stoch_ratio': 0.06,
                                                          'reshuffle': True})
        print('Preparing data for ThresherMediumTest...')
        medium_data = get_sample_data(path='./')

        self.scores = list(medium_data['pred'].values)
        self.actual_classes = list(medium_data['actual'].values)

        self.left_allowed, self.right_allowed = 0.40, 0.65

    def test_data_case(self):
        compute_result = self.t.optimize_threshold(self.scores, self.actual_classes)
        print(f'[ThresherMediumTest] Result found: {compute_result}')
        self.assertTrue(self.left_allowed <= compute_result < self.right_allowed,
                        msg="Checking proper result for the ThresherMediumTest")

    def test_data_case_alt(self):
        compute_result = self.alt_t.optimize_threshold(self.scores, self.actual_classes)
        print(f'[ThresherMediumTest][Alg:linear] Result found: {compute_result}')
        self.assertTrue(self.left_allowed <= compute_result < self.right_allowed,
                        msg="Checking proper result for the ThresherMediumTest")

    def test_data_case_alt2(self):
        compute_result = self.alt_t2.optimize_threshold(self.scores, self.actual_classes)
        print(f'[ThresherMediumTest][Alg:sim] Result found: {compute_result}')
        self.assertTrue(self.left_allowed <= compute_result < self.right_allowed,
                        msg="Checking proper result for the ThresherMediumTest")

    def test_data_case_alt3(self):
        compute_result = self.alt_t3.optimize_threshold(self.scores, self.actual_classes)
        print(f'[ThresherMediumTest][Alg:grid] Result found: {compute_result}')
        self.assertTrue(self.left_allowed <= compute_result < self.right_allowed,
                        msg="Checking proper result for the ThresherMediumTest")

    def test_data_case_alt4(self):
        compute_result = self.alt_t4.optimize_threshold(self.scores, self.actual_classes)
        print(f'[ThresherMediumTest][Alg:sgrid] Result found: {compute_result}')
        self.assertTrue(self.left_allowed <= compute_result < self.right_allowed,
                        msg="Checking proper result for the ThresherMediumTest")

    def test_data_case_alt5(self):
        compute_result = self.alt_t5.optimize_threshold(self.scores, self.actual_classes)
        print(f'[ThresherMediumTest][Alg:sgrid(/w shuffle)] Result found: {compute_result}')
        self.assertTrue(self.left_allowed <= compute_result < self.right_allowed,
                        msg="Checking proper result for the ThresherMediumTest")

    def test_oracle(self):
        self.assertTrue(run_oracle({'data_length': len(self.scores)}) == algorithm.available_algorithms['grid'])


class ThresherSmallTest(unittest.TestCase):

    def setUp(self):
        # Preparing data for unit test
        self.t = thresher.Thresher(progress_bar=False)
        self.scores = [0.1, 0.15, 0.2, 0.22, 0.27, 0.29, 0.3, 0.4, 0.7]

    def test_data_normalization(self):
        self.t = thresher.Thresher(labels=(0, 1))
        actual_classes = [0, 0, 0, 0, 0, 0, 0, 1, 1]
        compute_result = self.t.optimize_threshold(self.scores, actual_classes)
        print(f'[ThresherVerySmallTest] Result found: {compute_result}')
        self.assertTrue(0.3 <= compute_result < 0.4,
                        msg="Checking proper result for the ThresherVerySmallTest")

    def test_data_case(self):
        actual_classes = [-1, -1, -1, -1, -1, -1, -1, 1, 1]
        compute_result = self.t.optimize_threshold(self.scores, actual_classes)
        print(f'[ThresherVerySmallTest] Result found: {compute_result}')
        self.assertTrue(0.3 <= compute_result < 0.4,
                        msg="Checking proper result for the ThresherVerySmallTest")

    def test_data_case_parallel(self):
        self.t = thresher.Thresher(algorithm_params={'n_jobs': 3})
        actual_classes = [-1, -1, -1, -1, -1, -1, -1, 1, 1]
        compute_result = self.t.optimize_threshold(self.scores, actual_classes)
        print(f'[ThresherVerySmallTest] Result found: {compute_result}')
        self.assertTrue(0.3 <= compute_result < 0.4,
                        msg="Checking proper result for the ThresherVerySmallTest")


class ThresherVerySmallTest(unittest.TestCase):

    def setUp(self):
        # Preparing data for unit test
        self.t = thresher.Thresher(progress_bar=False)
        self.scores = [0.1, 0.3, 0.4, 0.7]

    def test_data_normalization(self):
        self.t = thresher.Thresher(labels=(0, 1))
        actual_classes = [0, 0, 1, 1]
        compute_result = self.t.optimize_threshold(self.scores, actual_classes)
        print(f'[ThresherVerySmallTest] Result found: {compute_result}')
        self.assertTrue(0.3 <= compute_result < 0.4,
                        msg="Checking proper result for the ThresherVerySmallTest")

    def test_data_case(self):
        actual_classes = [-1, -1, 1, 1]
        compute_result = self.t.optimize_threshold(self.scores, actual_classes)
        print(f'[ThresherVerySmallTest] Result found: {compute_result}')
        self.assertTrue(0.3 <= compute_result < 0.4,
                        msg="Checking proper result for the ThresherVerySmallTest")

    def test_options(self):
        self.assertTrue(len(self.t.get_supported_algorithms()) == 6,
                        msg="Checking if there are four available algorithms (including oracle)")


class ThresherCrashRegressionTest(unittest.TestCase):
    """Covers crashes fixed in 0.2.2.

    Every case here raised an exception before that release. They exercise paths the
    rest of the suite never reaches: algorithms selected explicitly rather than by the
    oracle, small inputs, and cleanly separable data.
    """

    @staticmethod
    def _separable(n):
        # Cleanly separable, which drives the stochastic evaluations to a zero
        # mis-classification ratio - the case that used to divide by zero in sgd.
        scores = [(i + 1) / (n + 1) for i in range(n)]
        return scores, [-1] * (n // 2) + [1] * (n - n // 2)

    def test_small_input_does_not_divide_by_zero(self):
        # int(stoch_ratio * N) floored to 0 below N=50 for 'gen' and N=20 for 'sgrid',
        # producing an empty sample.
        for algorithm in ('gen', 'sgrid', 'sgd'):
            for n in (4, 9, 19, 21, 45):
                with self.subTest(algorithm=algorithm, n=n):
                    scores, actual_classes = self._separable(n)
                    result = thresher.Thresher(algorithm=algorithm).optimize_threshold(scores, actual_classes)
                    self.assertIsInstance(result, float)

    def test_sgd_on_separable_data(self):
        # A perfect stochastic evaluation made 'previous_eval' 0.0, which the gradient
        # update then divided by.
        for n in (200, 500, 1000):
            with self.subTest(n=n):
                scores, actual_classes = self._separable(n)
                result = thresher.Thresher(algorithm='sgd').optimize_threshold(scores, actual_classes)
                self.assertIsInstance(result, float)

    def test_get_current_algorithm(self):
        # Used 'with' on an Algorithm namedtuple, so it raised TypeError unconditionally.
        t = thresher.Thresher(algorithm='grid')
        current = t.get_current_algorithm()
        self.assertEqual(current['name'], 'grid')
        self.assertEqual(current['object'], algorithm.available_algorithms['grid'])

    def test_linear_parallel_all_processors(self):
        # n_jobs=-1 is documented in the README, but made chunksize negative.
        scores, actual_classes = self._separable(200)
        t = thresher.Thresher(algorithm='linear', algorithm_params={'n_jobs': -1})
        result = t.optimize_threshold(scores, actual_classes)
        self.assertTrue(0.0 <= result <= 1.0)


class ThresherResultRangeTest(unittest.TestCase):
    """A returned threshold must lie within the range of the scores it was given.

    Anything outside it puts every sample in one class. 'sgd' used to walk out of that
    range on cleanly separable data and return e.g. 1.8972 for a predict_proba cut-off,
    which looks plausible enough to go unnoticed.
    """

    @staticmethod
    def _separable(n, seed):
        random.seed(seed)
        scores = sorted(random.random() for _ in range(n))
        return scores, [-1] * (n // 2) + [1] * (n - n // 2)

    def test_result_within_score_range(self):
        for alg in ('ls', 'sgd', 'gen', 'grid', 'sgrid'):
            for n in (200, 2000, 5000):
                with self.subTest(algorithm=alg, n=n):
                    scores, actual_classes = self._separable(n, seed=n)
                    result = thresher.Thresher(algorithm=alg).optimize_threshold(scores, actual_classes)
                    self.assertGreaterEqual(result, min(scores))
                    self.assertLessEqual(result, max(scores))

    def test_sgd_converges_near_the_optimum(self):
        # Guards the step-size cap: without it the walk overshoots, pins against a bound
        # and reports convergence there, landing far from the true threshold.
        for n in (2000, 5000):
            with self.subTest(n=n):
                scores, actual_classes = self._separable(n, seed=n)
                reference = thresher.Thresher(algorithm='ls').optimize_threshold(scores, actual_classes)
                result = thresher.Thresher(algorithm='sgd').optimize_threshold(scores, actual_classes)
                self.assertLess(abs(result - reference), 0.15)


class ThresherInputValidationTest(unittest.TestCase):
    """Covers the error reporting fixed in 0.2.3.

    Bad input previously surfaced as a bare StopIteration or a message-less
    AssertionError, neither of which told the caller what was actually wrong.
    """

    def test_unknown_algorithm_in_constructor(self):
        with self.assertRaises(ValueError) as ctx:
            thresher.Thresher(algorithm='does-not-exist')
        message = str(ctx.exception)
        self.assertIn('does-not-exist', message)
        # the message should list what the caller could have used instead
        for name in algorithm.available_algorithms:
            self.assertIn(name, message)

    def test_unknown_algorithm_in_set_algorithm(self):
        # This used to print a warning and silently keep the previous algorithm, so the
        # caller believed a switch had happened when it had not.
        t = thresher.Thresher(algorithm='grid')
        with self.assertRaises(ValueError):
            t.set_algorithm('does-not-exist')
        self.assertEqual(t.get_current_algorithm()['name'], 'grid')

    def test_known_aliases_still_resolve(self):
        for alias, expected in (('sim', 'gen'), ('genetic', 'gen'), ('linear', 'ls'),
                                ('gs', 'grid'), ('s-grid', 'sgrid'), ('curve_fitting', 'sgd')):
            with self.subTest(alias=alias):
                self.assertEqual(thresher.Thresher(algorithm=alias).get_current_algorithm()['name'], expected)

    def test_single_class_labels(self):
        with self.assertRaises(ValueError) as ctx:
            thresher.Thresher().optimize_threshold([0.1, 0.2, 0.3], [-1, -1, -1])
        self.assertIn('single class', str(ctx.exception))

    def test_unmapped_labels_point_at_the_labels_option(self):
        with self.assertRaises(ValueError) as ctx:
            thresher.Thresher().optimize_threshold([0.1, 0.2], [0, 1])
        self.assertIn('labels', str(ctx.exception))

    def test_empty_input(self):
        with self.assertRaises(ValueError):
            thresher.Thresher().optimize_threshold([], [])

    def test_validation_survives_optimized_mode(self):
        # The old check was an `assert`, which python -O strips entirely - malformed input
        # would then reach the solvers instead of being rejected.
        source = ('import thresher;'
                  'thresher.Thresher().optimize_threshold([0.1, 0.2, 0.3], [-1, -1, -1])')
        completed = subprocess.run([sys.executable, '-O', '-c', source],
                                   capture_output=True, text=True)
        self.assertNotEqual(completed.returncode, 0, msg='invalid input was accepted under -O')
        self.assertIn('ValueError', completed.stderr)


if __name__ == "__main__":
    print('Unit testing initiated. Running 4 different test cases, please wait....')
    unittest.main()
