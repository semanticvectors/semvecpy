from unittest import TestCase

import numpy.testing as npt

import vectors.vector_utils as vu
from vectors.vector_utils import cosine_similarity as cs
import vectors.graded_vectors as gv


class TestGradedVectors(TestCase):
    tol = 0.00001

    def test_graded_vector_factory(self):
        gvf = gv.GradedVectorFactory(100)
        for i in range(1, 11):
            self.assertLess(
                vu.cosine_similarity(gvf.get_vector_for_proportion(i/10.0), gvf.get_vector_for_proportion(0)),
                vu.cosine_similarity(gvf.get_vector_for_proportion((i-1)/10.0), gvf.get_vector_for_proportion(0)))

    def test_orthographic_vector_factory_similarities(self):
        ovf = gv.OrthographicVectorFactory(100)
        self.assertGreater(cs(ovf.get_vector("word"), ovf.get_vector("word")),
                           cs(ovf.get_vector("word"), ovf.get_vector("word2")))
        self.assertGreater(cs(ovf.get_vector("word"), ovf.get_vector("word2")),
                           cs(ovf.get_vector("word"), ovf.get_vector("word222222")))

    def test_orthographic_vector_factory_cache(self):
        ovf = gv.OrthographicVectorFactory(100)
        self.assertEqual(len(ovf.elemental_vectors), 0)
        self.assertEqual(len(ovf.word_vectors), 0)
        ovf.get_vector("word")
        self.assertEqual(len(ovf.elemental_vectors), 4)
        self.assertEqual(len(ovf.word_vectors), 1)
