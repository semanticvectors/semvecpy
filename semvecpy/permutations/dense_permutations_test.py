import unittest

import numpy as np
import numpy.testing as npt

from . import dense_permutations as dp


class TestDensePermutations(unittest.TestCase):
    tol = 0.00001

    def test_get_sort_permutation(self):
        vector = [0.3, 0.2, 0.4, 0.1]
        npt.assert_allclose([2, 0, 1, 3], dp.get_sort_permutation(vector), rtol=self.tol)

    def test_permute_vector(self):
        vector = np.asarray([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
        perm = [2, 3, 0, 1]
        answer = [0.3, 0.4, 0.1, 0.2]
        permuted = list(dp.permute_vector(perm, vector))
        npt.assert_allclose(answer, permuted, rtol=self.tol)

    def test_inverse_permutation(self):
        perm = [1, 4, 3, 0, 2]
        npt.assert_allclose([3, 0, 4, 2, 1], dp.inverse_permutation(perm), rtol=self.tol)

    def test_permutation_to_matrix(self):
        perm = np.asarray([1, 3, 2, 0])
        vector = np.asarray([0.1, 0.2, 0.3, 0.4])
        npt.assert_allclose(
            dp.permute_vector(perm, vector),
            # order of multiplication matters, vector first for the way the permutation matrix is set up
            np.matmul(vector, dp.permutation_to_matrix(perm)),
            rtol=self.tol)


if __name__ == '__main__':
    unittest.main()
