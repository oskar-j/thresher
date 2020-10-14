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
        self.assertTrue(run_oracle({'data_length': len(self.scores)}) == algorithm.available_algorithms['sgd'])


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


if __name__ == "__main__":
    print('Unit testing initiated. Running 4 different test cases, please wait....')
    unittest.main()
