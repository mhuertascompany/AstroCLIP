import unittest
import numpy as np
from euclid.umap_path import sample_segment


class PathTests(unittest.TestCase):
    def test_order_direction_and_filter(self):
        xy = np.array([[2, 0], [0, 0], [1, 0], [1, 9], [np.nan, 0]])
        self.assertEqual(sample_segment(xy, [0, 0], [2, 0], 3, .1), [1, 2, 0])
        self.assertEqual(sample_segment(xy, [2, 0], [0, 0], 3, .1), [0, 2, 1])
        self.assertEqual(sample_segment(xy, [0, 0], [2, 0], 3, .1,
                                       [True, True, False, True, True]), [1, 0])

    def test_sparse_unique_and_zero_length(self):
        xy = np.array([[0, 0], [10, 0]])
        self.assertEqual(sample_segment(xy, [0, 0], [10, 0], 11, .1), [0, 1])
        self.assertEqual(sample_segment(xy, [0, 5], [10, 5], 11, .1), [])
        with self.assertRaises(ValueError):
            sample_segment(xy, [0, 0], [0, 0])


if __name__ == '__main__':
    unittest.main()
