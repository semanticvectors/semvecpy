import os
import unittest

from . import semvec_utils as semvec


class TestSemvecUtils(unittest.TestCase):
    def setUp(self) -> None:
        # These few lines should enable the test setup to find the test data, wherever the test is run from.
        this_dir = os.path.dirname(__file__)
        semvecpy_root_dir = os.path.split(os.path.split(this_dir)[0])[0]
        test_data_dir = os.path.join(semvecpy_root_dir, "test_data")

        # vectors trained as follows:
        # java -cp semanticvectors-5.9.jar pitt.search.semanticvectors.ESP -luceneindexpath predication_index -vectortype binary -dimension 64 -trainingcycles 8 -mutablepredicatevectors
        self.predicate_vectors = semvec.readfile(os.path.join(test_data_dir, "predicatevectors.bin"))
        self.semantic_vectors = semvec.readfile(os.path.join(test_data_dir, "semanticvectors.bin"))
        self.elemental_vectors = semvec.readfile(os.path.join(test_data_dir, "elementalvectors.bin"))

    def test_compareterms1(self):
        result = semvec.compare_terms(term1="P(HAS_CURRENCY)", term2="P(CAPITAL_OF)",
                                      elemental_vectors=self.elemental_vectors, semantic_vectors=self.semantic_vectors,
                                      predicate_vectors=self.predicate_vectors)
        self.assertEqual(-0.0625, result)

    def test_compareterms2(self):
        result = semvec.compare_terms(term1="S(pretoria_(executive))", term2="E(south_africa)",
                                      elemental_vectors=self.elemental_vectors, semantic_vectors=self.semantic_vectors,
                                      predicate_vectors=self.predicate_vectors)
        self.assertEqual(-0.15625, result)

    def test_compareterms3(self):
        result = semvec.compare_terms(term1="S(pretoria_(executive))*E(south_africa)", term2="P(CAPITAL_OF)",
                                      elemental_vectors=self.elemental_vectors, semantic_vectors=self.semantic_vectors,
                                      predicate_vectors=self.predicate_vectors)
        self.assertEqual(0.84375, result)

    def test_compareterms4(self):
        with self.assertRaises(semvec.TermNotFoundError):
            semvec.compare_terms(term1="S(pretoria_(executive))*E(south_africa)", term2="P(not_a_term)",
                                 elemental_vectors=self.elemental_vectors, semantic_vectors=self.semantic_vectors,
                                 predicate_vectors=self.predicate_vectors)

    def test_compareterms5(self):
        with self.assertRaises(semvec.TermNotFoundError):
            semvec.compare_terms(term1="S(not_a_term)*E(south_africa)", term2="P(CAPITAL_OF)",
                                 elemental_vectors=self.elemental_vectors, semantic_vectors=self.semantic_vectors,
                                 predicate_vectors=self.predicate_vectors)

    def test_compareterms6(self):
        with self.assertRaises(semvec.TermNotFoundError):
            semvec.compare_terms(term1="S(pretoria_(executive))*E(not_a_term)", term2="P(CAPITAL_OF)",
                                 elemental_vectors=self.elemental_vectors, semantic_vectors=self.semantic_vectors,
                                 predicate_vectors=self.predicate_vectors)

    def test_compareterms7(self):
        with self.assertRaises(semvec.MalformedQueryError):
            semvec.compare_terms(term1="F(pretoria_(executive))*E(south_africa)", term2="P(CAPITAL_OF)",
                                 elemental_vectors=self.elemental_vectors, semantic_vectors=self.semantic_vectors,
                                 predicate_vectors=self.predicate_vectors)

    def test_comparetermsbatch1(self):
        terms = ["P(HAS_CURRENCY)|P(CAPITAL_OF)",
                 "S(pretoria_(executive))|E(south_africa)",
                 "S(pretoria_(executive))*E(south_africa)|P(CAPITAL_OF)"]
        result = semvec.compare_terms_batch(terms=terms, elemental_vectors=self.elemental_vectors,
                                            semantic_vectors=self.semantic_vectors,
                                            predicate_vectors=self.predicate_vectors)
        self.assertListEqual([-0.0625, -0.15625, 0.84375], result)

    def test_comparetermsbatch2(self):
        terms = []
        result = semvec.compare_terms_batch(terms=terms, elemental_vectors=self.elemental_vectors,
                                            semantic_vectors=self.semantic_vectors,
                                            predicate_vectors=self.predicate_vectors)
        self.assertListEqual([], result)

    def test_search1(self):
        result = semvec.search("S(pretoria_(executive))*E(south_africa)",
                               search_vectors=self.predicate_vectors,
                               elemental_vectors=self.elemental_vectors,
                               semantic_vectors=self.semantic_vectors,
                               predicate_vectors=self.predicate_vectors,
                               search_type="boundproduct")
        self.assertListEqual([
            [0.843750, "CAPITAL_OF"],
            [0.031250, "CAPITAL_OF-INV"],
            [0.000000, "HAS_CURRENCY-INV"],
            [-0.031250, "HAS_CURRENCY"],
            [-0.187500, "HAS_NATIONAL_ANIMAL"],
            [-0.406250, "HAS_NATIONAL_ANIMAL-INV"],
        ], result)

    def test_search2(self):
        result = semvec.search("CAPITAL_OF",
                               search_vectors=self.predicate_vectors,
                               elemental_vectors=self.elemental_vectors,
                               semantic_vectors=self.semantic_vectors,
                               predicate_vectors=self.predicate_vectors,
                               search_type="single_term")
        result2 = semvec.search("P(CAPITAL_OF)",
                               search_vectors=self.predicate_vectors,
                               elemental_vectors=self.elemental_vectors,
                               semantic_vectors=self.semantic_vectors,
                               predicate_vectors=self.predicate_vectors,
                               search_type="boundproduct")
        self.assertListEqual([
            [1.000000, "CAPITAL_OF"],
            [0.000000, "CAPITAL_OF-INV"],
            [-0.031250, "HAS_CURRENCY-INV"],
            [-0.062500, "HAS_CURRENCY"],
            [-0.218750, "HAS_NATIONAL_ANIMAL"],
            [-0.375000, "HAS_NATIONAL_ANIMAL-INV"],
        ], result)
        self.assertListEqual(result, result2)


if __name__ == '__main__':
    unittest.main()
