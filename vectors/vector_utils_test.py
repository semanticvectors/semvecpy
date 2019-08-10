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
