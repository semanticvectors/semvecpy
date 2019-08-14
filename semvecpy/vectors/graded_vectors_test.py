from unittest import TestCase

import numpy.testing as npt

from . import vector_utils as vu
from semvecpy.vectors.vector_utils import cosine_similarity as cs
from . import graded_vectors as gv


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
