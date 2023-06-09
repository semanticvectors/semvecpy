import os
import unittest
import numpy as np
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


    def test_pathfinder(self):
        #refernce pathfinder data
        rog_test = ['0',
                    '13',
                    '29 26',
                    '18 25 47',
                    '51 44 54 8',
                    '23 34 43 12 17',
                    '17 23 43 18 15 27',
                    '18 15 27 36 62 47 47',
                    '45 33 55 65 36 67 67 33',
                    '15 13 39 35 73 44 41 20 20',
                    '15 11 37 41 73 42 41 24 44 35',
                    '22 28 35 22 70 32 45 24 47 43 46',
                    '41 31 48 73 56 65 68 35 49 66 11 63',
                    '48 35 54 72 56 74 76 35 53 61 21 69 27',
                    '23 28 51 48 72 48 47 49 80 42 44 43 71 73',
                    '20 47 73 48 73 55 63 54 64 57 53 64 67 73 52',
                    '33 65 64 53 53 55 72 59 55 68 59 66 67 66 64 16',
                    '18 49 74 26 67 26 58 54 71 46 50 43 58 75 51 13 12',
                    '28 58 76 46 62 37 57 60 69 57 52 58 61 74 58 26 27 11',
                    '22 49 70 50 62 47 60 60 73 63 55 62 64 75 62 9 23 29 32',
                    '30 64 47 48 73 49 63 63 75 64 58 70 71 74 63 17 32 35 40 8',
                    '31 55 78 54 68 53 67 63 72 57 64 66 75 74 63 18 34 34 39 9 19',
                    '45 53 35 45 45 39 55 56 49 51 50 66 64 64 50 40 38 52 56 27 23 43',
                    '35 69 76 65 59 74 78 72 75 77 73 75 74 78 24 11 16 17 45 33 60 49 10',
                    '62 63 15 49 43 27 56 62 58 62 69 70 69 73 78 47 44 61 65 30 14 58 11 32']

        #different 'r' parameters too test
        rogrs = [1,1.01,1.05,1.1,1.15,1.2,1.4,1.6,1.8,2,3,4,5,6, np.inf]

        #reference results from canoonical Pathfinder implementationo
        rogref = [[	119	,	104	,	103	,	103	],
                    [	102	,	89	,	87	,	87	],
                    [	95	,	83	,	81	,	81	],
                    [	86	,	75	,	70	,	70	],
                    [	76	,	66	,	62	,	61	],
                    [	72	,	65	,	60	,	59	],
                    [	63	,	53	,	53	,	52	],
                    [	56	,	51	,	51	,	50	],
                    [	50	,	47	,	45	,	45	],
                    [	47	,	44	,	42	,	42	],
                    [	39	,	37	,	36	,	34	],
                    [	35	,	31	,	31	,	29	],
                    [	32	,	30	,	29	,	27	],
                    [	32	,	30	,	28	,	26	],
                    [	32	,	28	,	27	,	25	]]


        # fill in rest
        xs = [np.asarray(x.split(' '), dtype=int) for x in rog_test]
        xs = np.asarray(xs,dtype=object)
        nuxs = []
        for x in xs:
            toadd = xs.shape[0] - x.shape[0]
            nuxs.append(np.concatenate([np.asarray(x), np.zeros(toadd, dtype=int)]))

        nuxs = np.asarray(nuxs)
        #make symmetric
        nuxs = nuxs + nuxs.T - np.diag(np.diag(nuxs))
        #print(nuxs)

        #Test cases contributed by Roger Schvaneveldt
        #Test Pathfinder for all combinations four different 'q' parameter values
        #and 15 'r' parameter values, with reference results
        #from the canonical implementation

        for i, r in enumerate(rogrs):
            qs = [np.sum( semvec.pathfinder(q, r, nuxs, cosines=False) > 0) // 2 for q in  [2, 3, 4, 24]]
            np.testing.assert_almost_equal(qs,rogref[i])
            print('r', 'reference:',rogref[i],'this implementation:',qs, 'q of [2,3,4,24]')

        testfl = np.asarray([[1, 0.95, 0.24], [0.95, 1, 0.95], [0.24, 0.95, 1]])
        ansfl  = np.asarray([[1,  0.95, 0],[0.95, 1,  0.95],[0, 0.95 ,1 ]])
        pruned = semvec.pathfinder(8, 1, testfl);

        np.testing.assert_almost_equal(pruned,ansfl)

if __name__ == '__main__':
    unittest.main()

