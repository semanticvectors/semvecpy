from unittest import TestCase

import numpy.testing as npt

import vectors.vector_utils as vu
import vectors.real_vector_binding as rvb


class TestRealVectorBinding(TestCase):
    tol = 0.00001

    def test_circular_convolution(self):
        npt.assert_allclose([1, 1], rvb.circular_convolution([1, 0], [1, 1]),  rtol=self.tol)
        npt.assert_allclose([-1., 2., -1.], rvb.circular_convolution([0, -1, 1], [1, 2, 3]),  rtol=self.tol)

    def test_create_vector(self):
        vec = rvb.create_dense_random_vector(5, seed=2)
        npt.assert_allclose([-0.1280102, -0.94814754, 0.09932496, -0.12935521, -0.1592644], vec, rtol=self.tol)

    def test_graded_vector_factory(self):
        gvf = rvb.GradedVectorFactory(100)
        for i in range(1, 11):
            self.assertLess(
                vu.cosine_similarity(gvf.get_vector_for_proportion(i/10.0), gvf.get_vector_for_proportion(0)),
                vu.cosine_similarity(gvf.get_vector_for_proportion((i-1)/10.0), gvf.get_vector_for_proportion(0)))
