import unittest

import permutations.semvec_utils as semvec


class TestSemvecUtils(unittest.TestCase):
    def setUp(self) -> None:
        # vectors trained as follows:
        # java -cp semanticvectors-5.9.jar pitt.search.semanticvectors.ESP -luceneindexpath predication_index -vectortype binary -dimension 64 -trainingcycles 8 -mutablepredicatevectors
        self.predicate_vectors = semvec.readfile("../unittest_vectors/predicatevectors.bin")
        self.semantic_vectors = semvec.readfile("../unittest_vectors/semanticvectors.bin")
        self.elemental_vectors = semvec.readfile("../unittest_vectors/elementalvectors.bin")

    def test_compareterms1(self):
        result = semvec.compare_terms(term1="P(HAS_CURRENCY)", term2="P(CAPITAL_OF)",
                                      elemental_vectors=self.elemental_vectors, semantic_vectors=self.semantic_vectors,
                                      predicate_vectors=self.predicate_vectors)
        self.assertEqual(0.0625, result)

    def test_compareterms2(self):
        result = semvec.compare_terms(term1="S(pretoria_(executive))", term2="E(south_africa)",
                                      elemental_vectors=self.elemental_vectors, semantic_vectors=self.semantic_vectors,
                                      predicate_vectors=self.predicate_vectors)
        self.assertEqual(-0.03125, result)

    def test_compareterms3(self):
        result = semvec.compare_terms(term1="S(pretoria_(executive))*E(south_africa)", term2="P(CAPITAL_OF)",
                                      elemental_vectors=self.elemental_vectors, semantic_vectors=self.semantic_vectors,
                                      predicate_vectors=self.predicate_vectors)
        self.assertEqual(0.78125, result)

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
        self.assertListEqual([0.0625, -0.03125, 0.78125], result)

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
            [0.500000, "CAPITAL_OF"],
            [0.343750, "HAS_NATIONAL_ANIMAL"],
            [0.093750, "HAS_CURRENCY"],
            [0.062500, "HAS_NATIONAL_ANIMAL-INV"],
            [-0.093750, "HAS_CURRENCY-INV"],
            [-0.093750, "CAPITAL_OF-INV"]
        ], result)


if __name__ == '__main__':
    unittest.main()
