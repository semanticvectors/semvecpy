import math
import unittest

import numpy as np
import numpy.testing as npt

from . import vector_utils as vu


class TestVectorUtils(unittest.TestCase):
    tol = 0.0001

    def test_normalize(self):
        vector = np.asarray([1, 1, 1, 1], dtype=np.float32)
        npt.assert_allclose([0.5, 0.5, 0.5, 0.5], vu.normalize(vector), rtol=self.tol)

    def test_cosine_similarity(self):
        npt.assert_almost_equal(vu.cosine_similarity([1, 0], [0, 1]), 0)
        npt.assert_almost_equal(vu.cosine_similarity([1, 1], [0, 1]), math.sqrt(2)/2)

    def test_cosine_similarity_complex(self):
        npt.assert_almost_equal(vu.cosine_similarity([1j], [1j]), 1)
        npt.assert_almost_equal(vu.cosine_similarity([1j], [1 + 1j]), math.sqrt(2)/2)
        npt.assert_almost_equal(vu.cosine_similarity([1 + 1j, 0], [1j, 1]), 0.5)
        npt.assert_almost_equal(vu.cosine_similarity([1, 1j], [1j, 1]), 0)

    def test_circular_convolution(self):
        npt.assert_allclose([1, 1], vu.circular_convolution([1, 0], [1, 1]),  rtol=self.tol)
        npt.assert_allclose([-1., 2., -1.], vu.circular_convolution([0, -1, 1], [1, 2, 3]),  rtol=self.tol)

    def test_create_vector(self):
        vec = vu.create_dense_random_vector(5, seed=2)
        npt.assert_allclose([-0.1280102, -0.94814754, 0.09932496, -0.12935521, -0.1592644], vec, rtol=self.tol)

        # Do it again to be sure.
        vec2 = vu.create_dense_random_vector(5, seed=2)
        npt.assert_allclose(vec, vec2, rtol=self.tol)

        # Make sure a different seed gives a different answer.
        vec3 = vu.create_dense_random_vector(5, seed=3)
        npt.assert_allclose([ 0.101596,  0.416296, -0.418191,  0.021655,  0.785894], vec3, rtol=self.tol)

    def test_create_complex_vector(self):
        vec = vu.create_dense_random_vector(1, seed=2, field=np.complex)
        self.assertTrue(np.issubdtype(vec.dtype, np.complex128))
        self.assertFalse(np.issubdtype(vec.dtype, np.float64))
        npt.assert_almost_equal(-0.1280102 - 0.94814754j, vec[0], self.tol)

    def test_get_k_neighbors_from_pairs(self):
        pairs = [('x', (1, 0, 0)), ('y', (0, 1, 0)), ('z', (0, 0, 1))]
        nearest = vu.get_k_neighbors_from_pairs(pairs, (0.9, 0.2, 0.1), 2)
        self.assertListEqual([nearest[0][0], nearest[1][0]], ['x', 'y'])
        npt.assert_allclose([0.971, 0.216], [nearest[0][1], nearest[1][1]], rtol=0.01)

        # Check no overflow
        nearest = vu.get_k_neighbors_from_pairs(pairs, (0.9, 0.2, 0.1), 10)
        self.assertIsNotNone(nearest)

    def test_complex_normalize(self):
        vector = np.asarray([1, 1j, -1, -1j], dtype=np.complex)
        npt.assert_allclose([0.5, 0.5j, -0.5, -0.5j], vu.normalize(vector), rtol=self.tol)
