from unittest import TestCase
from . import real_vectors as rv
from . import semvec_utils as sv
from . import vector_utils as vu
import numpy as np
import os


class TestRealVectors(TestCase):


    def test_vector_store(self):
        # Borrowed from semvec_utils_test
        # These few lines should enable the test setup to find the test data, wherever the test is run from.
        this_dir = os.path.dirname(__file__)
        semvecpy_root_dir = os.path.split(os.path.split(this_dir)[0])[0]
        test_data_dir = os.path.join(semvecpy_root_dir, "test_data/real_vecs")
        vector_store = rv.RealVectorStore()
        vector_store.init_from_file(os.path.join(test_data_dir, "semanticvectors.bin"))
        vector_store2 = rv.RealVectorStore()
        termvecs=sv.readfile(os.path.join(test_data_dir, "semanticvectors.bin"))
        vector_store2.init_from_lists(termvecs[0],termvecs[1])
        # vectors trained as follows:
        # java -cp semanticvectors-5.9.jar pitt.search.semanticvectors.ESP -luceneindexpath ../predication_index -vectortype real -dimension 50 -seedlength 50 -trainingcycles 9
        for vecstore in [vector_store,vector_store2]:
            rvec = vecstore.get_vector('south_africa')
            self.assertEqual(rvec.get_dimension(),50)
            nearest = vecstore.knn(rvec,5)
            self.assertEqual(nearest[0][1], 'south_africa')
            nearest = vecstore.knn_term('south_africa', 5)
            self.assertEqual(nearest[0][1], 'south_africa')
            nearest = vecstore.knn_term('south_africa', 5, stdev=True)
            self.assertGreater(nearest[0][0],4)

        vector_store.write_vectors('tempvecs.bin')
        vector_store2.init_from_file('tempvecs.bin')
        self.assertAlmostEqual(vector_store.get_vector('south_africa').measure_overlap(vector_store2.get_vector('south_africa')),1,3)


    def test_superpose(self):
        rasevec = rv.RealVectorFactory.generate_zero_vector(50)
        toadd = rv.RealVectorFactory.generate_random_vector(50)
        toadd.normalize()
        rasevec.superpose(toadd,2)
        rasevec.normalize()
        overlap = rasevec.measure_overlap(toadd)
        self.assertAlmostEqual(overlap,1,3)

    def test_set(self):
        rasevec = rv.RealVectorFactory.generate_zero_vector(50)
        toadd = rv.RealVectorFactory.generate_random_vector(50)
        rasevec.set(toadd.vector.copy())
        overlap = vu.cosine_similarity(rasevec.vector,toadd.vector)
        self.assertAlmostEqual(overlap,1,3)

    def test_bind(self):
        rasevec = rv.RealVectorFactory.generate_random_vector(50)
        rasebind = rasevec.copy()
        tobind  = rv.RealVectorFactory.generate_random_vector(50)
        rasebind.bind(tobind)
        bound_overlap = vu.cosine_similarity(rasevec.vector, rasebind.vector)
        self.assertLess(bound_overlap, .5)
        rasebind.release(tobind)
        release_overlap = vu.cosine_similarity(rasevec.vector,rasebind.vector)
        self.assertGreater(release_overlap,.5)
        rasebind = rasevec.copy()
        rasebind.bind(tobind)
        rv.RealVector.exact_convolution_inverse = True
        rasebind.release(tobind)
        release_overlap = vu.cosine_similarity(rasevec.vector, rasebind.vector)
        rv.RealVector.exact_convolution_inverse = False
        self.assertAlmostEqual(release_overlap,1,3)

    def test_copy(self):
        rasevec = rv.RealVectorFactory.generate_random_vector(50)
        copyvec = rasevec.copy()
        copy_overlap = vu.cosine_similarity(rasevec.vector,copyvec.vector)
        self.assertAlmostEqual(copy_overlap,1,3)

    def test_normalize_all(self):
        counts=[]
        vecs =[]
        for i in range(10):
            rasevec = rv.RealVectorFactory.generate_random_vector(50)
            vecs.append(rasevec.vector)
            counts.append(str(i))

        vector_store = rv.RealVectorStore()
        vector_store.init_from_lists(counts, vecs)
        norms = np.sqrt((np.array(vector_store.vectors) * np.array(vector_store.vectors)).sum(axis=1))
        self.assertNotAlmostEqual(np.mean(norms),1,3)
        vector_store.normalize_all()
        norms = np.sqrt((np.array(vector_store.vectors) * np.array(vector_store.vectors)).sum(axis=1))
        self.assertAlmostEqual(np.mean(norms), 1, 3)
        self.assertAlmostEqual(np.linalg.norm(vector_store.real_vectors[0].vector),1,3)