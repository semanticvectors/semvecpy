from unittest import TestCase

import numpy as np
import numpy.testing as npt

from . import sparse_permutations as sp
from . import dense_permutations as dp


class TestSparsePermutations(TestCase):
    tol = 0.00001

    def test_get_sort_permutation(self):
        vector = [0.3, 0.2, 0.4, 0.1]
        npt.assert_allclose([2, 0, 1, 3], sp.get_sort_permutation(vector), rtol=self.tol)

    def test_inverse_permutation(self):
        perm = [1, 4, 3, 0, 2]
        npt.assert_allclose([3, 0, 4, 2, 1], sp.inverse_permutation(perm), rtol=self.tol)

    def test_permutation_to_matrix(self):
        perm = np.array([1, 3, 2, 0])
        vector = np.array([0.1, 0.2, 0.3, 0.4])
        npt.assert_allclose(
            dp.permute_vector(perm, vector),
            # order of multiplication matters, vector first for the way the permutation matrix is set up
            np.matmul(vector, dp.permutation_to_matrix(perm)), rtol=self.tol)
