"""
Binary vectors provide the basis for a representational approach
developed by Pentti Kanerva known as the Binary Spatter Code, with
the following key components:
(1) Randomly generated bit vectors with a .5 probability of a set bit in each component
(2) A *superposition* operator: this combines bit vectors elementwise via the
majority rule (more 1s than 0s: set to 1) with ties split at random
(3) A *binding* operator: elementwise exclusive or (XOR)

The implementation here uses some conventions developed in the Semantic Vectors Java package, specifically
maintainence of a "voting record" for each vector object, and the capacity to "tally votes" with or without
affecting this voting record. When the voting record is affected this is analogous to normalization.

For further details of the relationship between real, binary and complex ground fields as
a basis for Vector Symbolic Architectures, see:

- Widdows, D, Cohen, T, DeVine, L. Real, Complex, and Binary Semantic Vectors.
QI’12 Proceedings of the 6th International Symposium on Quantum Interactions. Paris, France; 2012.

For further discussion of Vector Symbolic Architectures in general, see:

- Gayler RW. Vector Symbolic Architectures answer Jackendoff’s challenges for cognitive neuroscience.
In: In Peter Slezak (Ed), ICCS/ASCS International Conference on Cognitive Science. Sydney, Australia. University of New South Wales.; 2004. p. 133–8.

- Kanerva P. Hyperdimensional computing: An introduction to computing in distributed representation
with high-dimensional random vectors. Cognitive Computation. 2009;1(2):139–159.

"""

import numpy as np
import struct
import multiprocessing
from multiprocess import Pool
from numpy import int8, int32
from numba import jit as njit
from scipy.stats import zscore
from bitarray import bitarray
from . import semvec_utils as svu




class BinaryVector(object):
    """
    BinaryVector objects include both a bit vector (bitset) and a voting record, and
    support the fundamental Vector-symbolic Architecture (see operations of binding and superposition.
    """
    # convert integer to number of 1 bits in its packed binary representation (for hamming distance)
    lookuptable = np.asarray([0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4, 1, 2, 2, 3, 2, 3,
                              3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4,
                              3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 1, 2,
                              2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5,
                              3, 4, 4, 5, 4, 5, 5, 6, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5,
                              5, 6, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 1, 2, 2, 3,
                              2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4,
                              4, 5, 4, 5, 5, 6, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
                              3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 2, 3, 3, 4, 3, 4,
                              4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6,
                              5, 6, 6, 7, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 4, 5,
                              5, 6, 5, 6, 6, 7, 5, 6, 6, 7, 6, 7, 7, 8], dtype=np.uint8)

    @staticmethod
    @njit(nopython=True, parallel=True)
    def hd(x, y, lookuptable):

        xord = x^y
        hd = np.empty_like(y)

        for i in np.arange(xord.shape[0]):
            hd[i] = lookuptable[xord[i]]
        return hd

    @staticmethod
    @njit(nopython=True, parallel=True)
    def getcount_numba(x,lookuptable):
        xord = x
        hd = np.empty_like(x)
        for i in np.arange(xord.shape[0]):
            hd[i] = lookuptable[xord[i]]
        return np.sum(hd)

    @staticmethod
    def getcount(x):
        return BinaryVector.getcount_numba(x,BinaryVector.lookuptable)

    def __init__(self, dimension, incoming_vector=None, allposrecs,allnegrecs,allveclinks, index):
        self.dimension = dimension
        if incoming_vector is None:
            self.bitset = np.zeros(int(dimension/8),dtype=np.uint8)
        else:
            self.bitset = incoming_vector

        self.pvr = allposrecs[index]
        self.nvr = allnegrecs[index]
        allveclinks[index] = -1
        self.cv  = None
        self.ev  = None

    def voting_record_to_bitset(self,pvrs,nvrs,links,index):
        """
        Initializes the voting record and matches this to the current bitset. In many cases this
        won't be necessary, as a voting record is only required for superposition, and
        takes up 32x the RAM of the bitset.
        """
        if self.bitset.count(True) > 0:
            pvrs[index] = self.bitset#np.packbits(self.bitset))
            #self.nvr.append(np.packbits(bitarray('0' * len(self.bitset))))
            dimstring = '>' + str(int(len(self.bitset) * 0.125)) + 'B'
            nvrs[index] = struct.unpack(dimstring, bitarray('0' * len(self.bitset)).tobytes())
            links[index] = -1

    def set(self, incoming_bitarray):
        """
        Sets the bit vector and voting record to the incoming bitarray (not a copy)
        :param incoming_bitarray: a bitarray.bitarray object
        """
        self.bitset = np.packbits(incoming_bitarray)



    # Non negative overlap, normalized for dimensions
    def measure_overlap(self, other):
        """
        Measures the overlap between binary vectors as max(1 - (2.HD/d), 0)
        where d=dimensionality and HD=Hamming Distance.
        (HD/d of .5 defines orthogonality in the Binary Spatter Code)
        :param other: another BinaryVector
        :return the non-negative normalized hamming distance (NNHD - see above)
        """
        nhd = np.sum(BinaryVector.hd(self.bitset,other.bitset,BinaryVector.lookuptable))
        nhd = 2 * (0.5 - np.sum(nhd) / self.dimension)
        return nhd

    def addvector(self, cv, vrs,index,links):
        """
        Basic addition operation - adds a bitset to a voting record with a weight of one
        :param cv:
        :param vr:
        :return:
        """

        previndex = index

        while index >= 0:
            vrs[index] = vrs[ndex] ^ cv
            cv = cv & ~vrs[index]
            previndex = index
            index = links[index]
        if BinaryVector.getcount(cv) > 0:
            newindex = np.max(links) + 1
            links[previndex] = newindex
            vrs[newindex] = cv

    def addfromfloor(self, cv, vrs, index, links, rowfloor):
        """
       Faster addition operation for higher weights - adds a bitset to a voting record starting
       at the row corresponding to log2(weight), with higher rows representing higher numbers
       (row n represent the nth lowest order bit of a binary number)
       :param cv:
       :param vr:
       :return:
       """


        for i in range(rowfloor):
            previndex = index
            index=links[index]

        while index >= 0:
            vrs[index] = vrs[index] ^ cv
            cv = cv & ~vrs[index]
            previndex = index
            index=links[index]
            #todo - check loop structure - what would happen if the intervening voting records don't yet exist?
        if BinaryVector.getcount(cv) > 0: #cv.count(True) > 0:
            newindex = np.max(links) + 1
            links[previndex] = newindex
            vrs[newindex] = cv

    def reduce(self, pvrs,nvrs,plinks,nlinks,index):
        """
        Eliminate unecessary bits to save space in two steps:
            (1) positive and negative records cancel out
            (2) higher order rows that are identical (both zero at this point) are removed
        """
        """
        :return: 
        """
        pindex = index
        indextrace = []

        while pindex != -1:
            indextrace.append(pindex)
            c = self.nvr[pinex].copy()
            self.nvr[pindex] = self.nvr[i] & ~self.pvr[pindex]
            self.pvr[pindex] = self.pvr[pindex] & ~c
            pindex = links[pindex]

        nindex = index

        for i in range(len(pindextrace) - 1, -1, -1):
            pi = pindextrace[i]
            if (self.pvr[pi] ^ self.nvr[pi]).count() == 0:
                del self.pvr[pi]
                del self.nvr[pi]
                links[pi] = -1
            else:
                break

    def superpose(self, other, weight):
        """
        Adds the bit vector of the incoming BinaryVector 'other' to the voting record of self
        (in this case to the positive and negative voting records)
        :param other: another BinaryVector
        """

        if isinstance(weight, float):
            weight = int(weight * 100)

        if weight == 0:
            return

        if weight < 0:
            other = other.copy()
            other.bitset = ~other.bitset #.invert()
            weight = np.abs(weight)


        # decompose into powers of two, start at the highest possible level
        # e.g. if adding with a weight of eight, start in the fourth row
        rowfloor = int(np.floor(np.log2(weight)))
        while 0 < rowfloor < len(self.pvr) and weight > 0:
            weight = weight - int(np.power(2, rowfloor))
            self.cv = other.bitset.copy()
            self.addfromfloor(self.cv, self.pvr, rowfloor)
            self.cv = other.bitset.copy()
            self.cv = ~self.cv #.invert()
            self.addfromfloor(self.cv, self.nvr, rowfloor)
            if weight > 0:
                rowfloor = int(np.floor(np.log2(weight)))

        for q in range(weight):  # incrementally add the rest
            self.cv = other.bitset.copy()
            self.addvector(self.cv, self.pvr)
            self.cv = other.bitset.copy()
            self.cv = ~self.cv #.invert()
            self.addvector(self.cv, self.nvr)

        # ensure same length
        self.cv = self.cv ^ self.cv
        while len(self.nvr) < len(self.pvr):
            self.nvr.append(self.cv.copy())
        while len(self.pvr) < len(self.nvr):
            self.pvr.append(self.cv.copy())

    def bind(self, other):
        """
        Binds the bit vector of the incoming BinaryVector 'other' to the bit vector of self
        :param other: another BinaryVector
        """
        self.bitset ^= other.bitset

    def release(self, other):
        """
        The inverse of the bind operator, which with bit vectors is also pairwise XOR (a self-inverse)
        :param other: another BinaryVector
        """
        self.bitset ^= other.bitset

    def tally_votes(self, pvrs,nvrs, links, index):
        """
        Sets self bit vector (bitset) to the outcome of tallying the votes of the voting record
        (more 1s than zeros : 1, with ties split at random)
        """
        if pvrs[index] == -1: #not quite
            return  # nothing to tally

        self.cv = np.packbits(np.zeros(len(self.pvr[0])*8).astype(bool))
        self.ev = np.packbits(np.ones(len(self.pvr[0])*8).astype(bool))


        for i in range(0, len(self.pvr)):
            self.cv = self.cv | (self.pvr[i] & ~self.nvr[i])
            self.cv = self.cv & ~ (self.nvr[i] & ~self.pvr[i])
            self.ev = self.ev & ~(self.pvr[i] ^ self.nvr[i])

        self.randvec = BinaryVectorFactory.generate_random_vector(len(self.pvr[0])*8).bitset
        self.ev = self.ev & self.randvec
        self.cv = self.cv | self.ev
        self.bitset = self.cv

    def normalize(self):
        """
        Tallies votes and resets the voting record to None
        """
        self.tally_votes()
        self.pvr = np.array
        self.nvr = np.array


