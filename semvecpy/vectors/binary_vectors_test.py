from unittest import TestCase
from semvecpy.vectors import binary_vectors as bv
import numpy as np


class TestBinaryVectors(TestCase):

    def test_generate_random(self):
        meancard=[]
        for i in range(1,100):
            binvec = bv.BinaryVectorFactory.generate_random_vector(self,512)
            meancard.append(binvec.bitset.count(True))

        diffcard=np.average(meancard) - 256
        self.assertEqual(diffcard, 0)

    def test_generate_zero(self):
        allcards = []
        for i in range(1, 100):
            binvec = bv.BinaryVectorFactory.generate_zero_vector(self,512)
            allcards.append(binvec.bitset.count(True))

        sumcard = np.sum(allcards)
        self.assertEqual(sumcard, 0)

    def test_superpose(self):
        basevec = bv.BinaryVectorFactory.generate_zero_vector(self,512)
        toadd = bv.BinaryVectorFactory.generate_random_vector(self,512)
        basevec.superpose(toadd,1)
        basevec.tallyVotes()
        overlap = basevec.measure_overlap(toadd)
        self.assertEqual(overlap,1)

    def test_bind(self):
        basevec = bv.BinaryVectorFactory.generate_random_vector(self,512)
        basebind = basevec.copy()
        tobind  = bv.BinaryVectorFactory.generate_random_vector(self, 512)
        basebind.bind(tobind)
        bound_overlap = basebind.measure_overlap(basevec)
        self.assertAlmostEqual(bound_overlap,0,3)
        basebind.release(tobind)
        release_overlap = basebind.measure_overlap(basevec)
        self.assertEqual(release_overlap,1)