from unittest import TestCase
from . import binary_bitvectors as bv
from . import semvec_utils as sv
import numpy as np
import os


class TestBinaryVectors(TestCase):

    def test_vector_store(self):
        # Borrowed from semvec_utils_test
        # These few lines should enable the test setup to find the test data, wherever the test is run from.
        this_dir = os.path.dirname(__file__)
        semvecpy_root_dir = os.path.split(os.path.split(this_dir)[0])[0]
        test_data_dir = os.path.join(semvecpy_root_dir, "test_data")
        vector_store = bv.BinaryVectorStore()
        vector_store.init_from_file(os.path.join(test_data_dir, "semanticvectors.bin"))
        vector_store2 = bv.BinaryVectorStore()
        termvecs  = sv.readfile(os.path.join(test_data_dir, "semanticvectors.bin"))
        vector_store2.init_from_lists(termvecs[0],termvecs[1])
        # vectors trained as follows:
        # java -cp semanticvectors-5.9.jar pitt.search.semanticvectors.ESP -luceneindexpath predication_index -vectortype binary -dimension 64 -trainingcycles 8 -mutablepredicatevectors
        for vecstore in [vector_store,vector_store2]:
            bvec = vecstore.get_vector('south_africa')
            self.assertEqual(bvec.get_dimension(),64)
            nearest = vecstore.knn(bvec,5)
            self.assertEqual(nearest[0][1], 'south_africa')
            nearest = vecstore.knn_term('south_africa', 5)
            self.assertEqual(nearest[0][1], 'south_africa')
            nearest = vecstore.knn_term('south_africa', 5, stdev=True)
            self.assertGreater(nearest[0][0], 4)

        vector_store.write_vectors('tempvecs.bin')
        vector_store2.init_from_file('tempvecs.bin')
        self.assertAlmostEqual(
            vector_store.get_vector('south_africa').measure_overlap(vector_store2.get_vector('south_africa')), 1, 3)

    def test_generate_random(self):
        meancard=[]
        for i in range(1,100):
            binvec = bv.BinaryVectorFactory.generate_random_vector(512)
            meancard.append(binvec.bitset.count(True))

        diffcard=np.average(meancard) - 256
        self.assertEqual(diffcard, 0)

    def test_generate_zero(self):
        allcards = []
        for i in range(1, 100):
            binvec = bv.BinaryVectorFactory.generate_zero_vector(512)
            allcards.append(binvec.bitset.count(True))

        sumcard = np.sum(allcards)
        self.assertEqual(sumcard, 0)

    def test_superpose(self):
        basevec = bv.BinaryVectorFactory.generate_zero_vector(512)
        toadd = bv.BinaryVectorFactory.generate_random_vector(512)
        basevec.superpose(toadd,1)
        basevec.tally_votes()
        overlap = basevec.measure_overlap(toadd)
        self.assertEqual(overlap,1)
        toadd.superpose(toadd,.5)
        toadd.normalize()
        overlap = basevec.measure_overlap(toadd)
        self.assertEqual(overlap, 1)


    def test_set(self):
        basevec = bv.BinaryVectorFactory.generate_zero_vector(512)
        toadd = bv.BinaryVectorFactory.generate_random_vector(512)
        basevec.set(toadd.bitset.copy())
        overlap = basevec.measure_overlap(toadd)
        basevec.tally_votes()
        overlap += basevec.measure_overlap(toadd)
        self.assertEqual(overlap,2)

    def test_bind(self):
        basevec = bv.BinaryVectorFactory.generate_random_vector(512)
        basebind = basevec.copy()
        tobind  = bv.BinaryVectorFactory.generate_random_vector(512)
        basebind.bind(tobind)
        bound_overlap = basebind.measure_overlap(basevec)
        self.assertAlmostEqual(bound_overlap,0,3)
        basebind.release(tobind)
        release_overlap = basebind.measure_overlap(basevec)
        self.assertEqual(release_overlap,1)
