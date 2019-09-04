import math
import numpy as np
import numpy.testing as npt

from unittest import TestCase

from . import vector_utils as vu
from .vector_utils import cosine_similarity as cs
from . import graded_vectors as gv


class TestGradedVectors(TestCase):
    tol = 0.00001

    def test_gvf_complex_interpolation(self):
        gvf = gv.GradedVectorFactory(1, field=np.complex)
        gvf.alpha_vec = np.array([1])
        gvf.omega_vec = np.array([1j])
        ave = gvf.get_vector_for_proportion(0.5)
        npt.assert_allclose([math.sqrt(2)/2*(1 + 1j)], ave, rtol=self.tol)
        npt.assert_almost_equal(math.sqrt(2)/2, vu.cosine_similarity(gvf.alpha_vec, ave))
        npt.assert_almost_equal(math.sqrt(2)/2, vu.cosine_similarity(gvf.omega_vec, ave))

    def test_graded_vector_factory_real(self):
        gvf = gv.GradedVectorFactory(100)
        for i in range(1, 11):
            self.assertLess(
                vu.cosine_similarity(
                    gvf.get_vector_for_proportion(i/10.0), gvf.get_vector_for_proportion(0)),
                vu.cosine_similarity(
                    gvf.get_vector_for_proportion((i-1)/10.0), gvf.get_vector_for_proportion(0)))

    def test_graded_vector_factory_complex(self):
        gvf = gv.GradedVectorFactory(100, field=np.complex)
        for i in range(1, 11):
            self.assertLess(
                np.absolute(vu.cosine_similarity(
                    gvf.get_vector_for_proportion(i/10.0), gvf.get_vector_for_proportion(0))),
                np.absolute(vu.cosine_similarity(
                    gvf.get_vector_for_proportion((i-1)/10.0), gvf.get_vector_for_proportion(0))))

    def test_orthographic_vector_factory_similarities(self):
        ovf = gv.OrthographicVectorFactory(100)
        self.assertGreater(cs(ovf.get_word_vector("word"), ovf.get_word_vector("word")),
                           cs(ovf.get_word_vector("word"), ovf.get_word_vector("word2")))
        self.assertGreater(cs(ovf.get_word_vector("word"), ovf.get_word_vector("word2")),
                           cs(ovf.get_word_vector("word"), ovf.get_word_vector("word222222")))

    def test_orthographic_vector_factory_similarities_complex(self):
        ovf = gv.OrthographicVectorFactory(100, field=np.complex)
        self.assertGreater(cs(ovf.get_word_vector("word"), ovf.get_word_vector("word")),
                           cs(ovf.get_word_vector("word"), ovf.get_word_vector("word2")))
        self.assertGreater(cs(ovf.get_word_vector("word"), ovf.get_word_vector("word2")),
                           cs(ovf.get_word_vector("word"), ovf.get_word_vector("word222222")))

    def test_orthographic_vector_factory_cache(self):
        ovf = gv.OrthographicVectorFactory(100)
        self.assertEqual(len(ovf.character_vectors), 0)
        self.assertEqual(len(ovf.word_vectors), 0)
        ovf.get_word_vector("word")
        self.assertEqual(len(ovf.character_vectors), 4)
        self.assertEqual(len(ovf.word_vectors), 1)

    def test_orthographic_vector_factory_search(self):
        ovf = gv.OrthographicVectorFactory(100)
        for word in ["boo", "bit", "bot"]:
            ovf.make_and_store_word_vector(word)
        results = ovf.get_k_nearest_neighbors("boo", 3)
        self.assertListEqual([result[0] for result in results], ["boo", "bot", "bit"])
