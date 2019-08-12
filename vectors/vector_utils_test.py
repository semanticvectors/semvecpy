import math
import unittest

import numpy as np
import numpy.testing as npt

import vectors.vector_utils as vu


class TestPermutations(unittest.TestCase):
    tol = 0.000001

    def test_normalize(self):
        vector = np.asarray([1, 1, 1, 1], dtype=np.float32)
        npt.assert_allclose([0.5, 0.5, 0.5, 0.5], vu.normalize(vector).tolist(), rtol=self.tol)

    def test_cosine_similarity(self):
        npt.assert_almost_equal(vu.cosine_similarity([1, 0], [0, 1]), 0)
        npt.assert_almost_equal(vu.cosine_similarity([1, 1], [0, 1]), math.sqrt(2)/2)

    def test_circular_convolution(self):
        npt.assert_allclose([1, 1], vu.circular_convolution([1, 0], [1, 1]),  rtol=self.tol)
        npt.assert_allclose([-1., 2., -1.], vu.circular_convolution([0, -1, 1], [1, 2, 3]),  rtol=self.tol)

    def test_create_vector(self):
        vec = vu.create_dense_random_vector(5, seed=2)
        npt.assert_allclose([-0.1280102, -0.94814754, 0.09932496, -0.12935521, -0.1592644], vec, rtol=self.tol)
