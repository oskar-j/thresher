import unittest
import thresher


class ThresherVerySmallTest(unittest.TestCase):

    def setUp(self):
        # Preparing data for unit test
        self.t = thresher.Thresher()

    def test_data_case(self):
        scores = [0.1, 0.3, 0.4, 0.7]
        actual_classes = [-1, -1, 1, 1]
        compute_result = self.t.optimize_threshold(scores, actual_classes)
        self.assertTrue(compute_result == 0.3,
                        msg="Checking proper result for the ThresherVerySmallTest")

    def test_options(self):
        self.assertTrue(len(self.t.get_supported_algorithms()) == 4,
                        msg="Checking if there are four available algorithms")


if __name__ == "__main__":
    print('Unit testing initiated. Running 4 different test cases, please wait....')
    unittest.main()
