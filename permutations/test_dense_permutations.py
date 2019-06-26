import unittest

import numpy as np

import dense_permutations as p

"""
TODO:

Migrate tests to np.testing, as this will in general deal with floating
point issues.
"""


class TestPermutations(unittest.TestCase):
    def test_normalize(self):
        vector = np.asarray([1, 1, 1, 1], dtype=np.float32)
        #p.normalize(vector)
        #self.assertListEqual([0.5, 0.5, 0.5, 0.5], vector)
        self.assertListEqual([0.5, 0.5, 0.5, 0.5], list(p.normalize(vector).tolist()))

    def test_get_sort_permutation(self):
        vector = [0.3, 0.2, 0.4, 0.1]
        self.assertListEqual([2, 0, 1, 3], list(p.get_sort_permutation(vector)))

    def test_permute_vector(self):
        vector = np.asarray([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
        perm = [2, 3, 0, 1]
        answer = [0.3, 0.4, 0.1, 0.2]
        permuted = list(p.permute_vector(perm, vector))
        for i in range(4):
            self.assertAlmostEqual(answer[i], permuted[i])

    def test_inverse_permutation(self):
        perm = [1, 4, 3, 0, 2]
        self.assertListEqual([3, 0, 4, 2, 1], list(p.inverse_permutation(perm)))

    def test_permutation_to_matrix(self):
        perm = np.asarray([1, 3, 2, 0])
        vector = np.asarray([0.1, 0.2, 0.3, 0.4])
        self.assertListEqual(
            list(p.permute_vector(perm, vector)),
            list(np.matmul(vector, p.permutation_to_matrix(perm))) # order of multiplication matters, vector first for the way the permutation matrix is set up
            )

if __name__ == '__main__':
    unittest.main()