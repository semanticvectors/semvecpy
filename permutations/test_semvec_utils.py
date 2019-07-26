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
        result = semvec.compare_terms(predicate_vectors=self.predicate_vectors,
                                      semantic_vectors=self.semantic_vectors,
                                      elemental_vectors=self.elemental_vectors,
                                      term1="P(HAS_CURRENCY)", term2="P(CAPITAL_OF)")
        self.assertEqual(0.0625, result)

    def test_compareterms2(self):
        result = semvec.compare_terms(predicate_vectors=self.predicate_vectors,
                                      semantic_vectors=self.semantic_vectors,
                                      elemental_vectors=self.elemental_vectors,
                                      term1="S(pretoria_(executive))", term2="E(south_africa)")
        self.assertEqual(-0.03125, result)

    def test_compareterms3(self):
        result = semvec.compare_terms(predicate_vectors=self.predicate_vectors,
                                      semantic_vectors=self.semantic_vectors,
                                      elemental_vectors=self.elemental_vectors,
                                      term1="S(pretoria_(executive))*E(south_africa)", term2="P(CAPITAL_OF)")
        self.assertEqual(0.78125, result)

    def test_compareterms4(self):
        with self.assertRaises(semvec.TermNotFoundError):
            semvec.compare_terms(predicate_vectors=self.predicate_vectors,
                                 semantic_vectors=self.semantic_vectors,
                                 elemental_vectors=self.elemental_vectors,
                                 term1="S(pretoria_(executive))*E(south_africa)",
                                 term2="P(not_a_term)")

    def test_compareterms5(self):
        with self.assertRaises(semvec.TermNotFoundError):
            semvec.compare_terms(predicate_vectors=self.predicate_vectors,
                                 semantic_vectors=self.semantic_vectors,
                                 elemental_vectors=self.elemental_vectors,
                                 term1="S(not_a_term)*E(south_africa)",
                                 term2="P(CAPITAL_OF)")

    def test_compareterms6(self):
        with self.assertRaises(semvec.TermNotFoundError):
            semvec.compare_terms(predicate_vectors=self.predicate_vectors,
                                 semantic_vectors=self.semantic_vectors,
                                 elemental_vectors=self.elemental_vectors,
                                 term1="S(pretoria_(executive))*E(not_a_term)",
                                 term2="P(CAPITAL_OF)")

    def test_compareterms7(self):
        with self.assertRaises(semvec.MalformedQueryError):
            semvec.compare_terms(predicate_vectors=self.predicate_vectors,
                                 semantic_vectors=self.semantic_vectors,
                                 elemental_vectors=self.elemental_vectors,
                                 term1="F(pretoria_(executive))*E(south_africa)",
                                 term2="P(CAPITAL_OF)")

    def test_comparetermsbatch1(self):
        terms = ["P(HAS_CURRENCY)|P(CAPITAL_OF)",
                 "S(pretoria_(executive))|E(south_africa)",
                 "S(pretoria_(executive))*E(south_africa)|P(CAPITAL_OF)"]
        result = semvec.compare_terms_batch(predicate_vectors=self.predicate_vectors,
                                            semantic_vectors=self.semantic_vectors,
                                            elemental_vectors=self.elemental_vectors,
                                            terms=terms)
        self.assertListEqual([0.0625, -0.03125, 0.78125], result)

    def test_comparetermsbatch2(self):
        terms = []
        result = semvec.compare_terms_batch(predicate_vectors=self.predicate_vectors,
                                            semantic_vectors=self.semantic_vectors,
                                            elemental_vectors=self.elemental_vectors,
                                            terms=terms)
        self.assertListEqual([], result)


if __name__ == '__main__':
    unittest.main()
