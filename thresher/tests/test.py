import unittest
import thresher
from thresher import algorithm
from thresher.oracle import run_oracle
from thresher.tests.sample_data import get_sample_data


class ThresherMediumTest(unittest.TestCase):

    def setUp(self):
        # Preparing data for unit test
        self.t = thresher.Thresher(verbose=False, progress_bar=False)
        medium_data = get_sample_data(path='./')
        self.scores = list(medium_data['pred'].values)
        self.actual_classes = list(medium_data['actual'].values)

    def test_data_case(self):
        compute_result = self.t.optimize_threshold(self.scores, self.actual_classes)
        print(f'[ThresherMediumTest] Result found: {compute_result}')
        self.assertTrue(0.4 <= compute_result < 0.6,
                        msg="Checking proper result for the ThresherMediumTest")

    def test_oracle(self):
        self.assertTrue(run_oracle({'data_length': len(self.scores)}) == algorithm.available_algorithms['sgd'])


class ThresherVerySmallTest(unittest.TestCase):

    def setUp(self):
        # Preparing data for unit test
        self.t = thresher.Thresher(progress_bar=False)

    def test_data_case(self):
        scores = [0.1, 0.3, 0.4, 0.7]
        actual_classes = [-1, -1, 1, 1]
        compute_result = self.t.optimize_threshold(scores, actual_classes)
        print(f'[ThresherVerySmallTest] Result found: {compute_result}')
        self.assertTrue(0.3 <= compute_result < 0.4,
                        msg="Checking proper result for the ThresherVerySmallTest")

    def test_options(self):
        self.assertTrue(len(self.t.get_supported_algorithms()) == 4,
                        msg="Checking if there are four available algorithms")


if __name__ == "__main__":
    print('Unit testing initiated. Running 4 different test cases, please wait....')
    unittest.main()
