from unittest import TestCase

import numpy as np

import sparse_permutations as p


class TestPermutations(TestCase):
    def test_normalize(self):
        vector = [1, 1, 1, 1]
        p.normalize(vector)
        self.assertListEqual([0.5, 0.5, 0.5, 0.5], vector)

    def test_get_sort_permutation(self):
        vector = [0.3, 0.2, 0.4, 0.1]
        self.assertListEqual([2, 0, 1, 3], p.get_sort_permutation(vector))

    def test_permute_vector(self):
        vector = [0.1, 0.2, 0.3, 0.4]
        perm = [2, 3, 0, 1]
        self.assertListEqual([0.3, 0.4, 0.1, 0.2], list(p.permute_vector(perm, vector)))

    def test_inverse_permutation(self):
        perm = [1, 4, 3, 0, 2]
        self.assertListEqual([3, 0, 4, 2, 1], p.inverse_permutation(perm))

    def test_permutation_to_matrix(self):
        perm = [1, 3, 2, 0]
        vector = [0.1, 0.2, 0.3, 0.4]
        self.assertListEqual(
            list(p.permute_vector(perm, vector)),
            list(np.matmul(p.permutation_to_matrix(perm), vector)))
