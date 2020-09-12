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


class BinaryVectorFactory:
    """
    BinaryVectorFactory creates three sorts of BinaryVector objects:
    (1) Zero vectors (binary vectors with no bits set)
    (2) Random vectors (binary vectors with half the bits set to 1 at random)
    (3) Vectors with a preset bitarray (e.g. read from disk)
    """

    @staticmethod
    def generate_random_vector(dimension):
        randvec = BinaryVector(dimension)
        randvec.set_random_vector()
        return randvec

    @staticmethod
    def generate_zero_vector(dimension):
        zerovec = BinaryVector(dimension)
        return zerovec

    @staticmethod
    def generate_vector(incoming_packedbits):
        binaryvec = BinaryVector(len(incoming_packedbits)*8,incoming_vector=incoming_packedbits)
        return binaryvec


class BinaryVectorStore(object):
    """
    Storage, retrieval and nearest neighbor search of binary vectors
    """


    def __init__(self):
        self.dict = {}
        self.vectors = []
        self.bitvectors = np.ndarray
        self.terms = []

    def init_from_lists(self,terms,bitarrays):
        """
        Initializes from lists (e.g. those used by semvec_utils). Bitarray retained here
        for interoperability with other bit vector implementations.
        :param terms: list of terms
        :param binary_vectors: list of binary vectors
        """
        self.terms=terms
        self.vectors = [BinaryVectorFactory.generate_vector(np.packbits(args)) for args in bitarrays]
        self.bitvectors = np.asarray([args.bitset for args in self.vectors],
                                     dtype=np.uint8)

        self.dict = dict(zip(self.terms, self.vectors))



    def init_from_file(self,file_name):
        """
        Reads bit vectors from disk (Semantic Vectors binary format) into BinaryVector objects
        Creates a dictionary for retrieval
        :param filename:
        """
        self.terms = []
        self.bitvectors = []
        with open(file_name, mode='rb') as file:  # b is important -> binary
            file_content = file.read(1)
            x = file_content
            ct = int.from_bytes(x, byteorder='little', signed=False)
            file_content = file.read(ct)
            header = file_content.decode().split(" ")
            vindex = header.index('-vectortype')
            vectortype = header[vindex + 1]
            dindex = header.index('-dimension')
            dimension = int(header[dindex + 1])
            if vectortype != 'BINARY':
                print('Can\'t initialize binary vector store from ',vectortype,' vectors.')
                return

            unitsize = .125
            dimstring = '>' +str(int(dimension*unitsize)) + 'B'

            file_content = file.read(1)
            while file_content:
                # y = int.from_bytes(file_content[ct:ct + 1], byteorder='little', signed=False)

                # Read Lucene's vInt - if the most significant bit
                # is set, read another byte as significant bits
                # ahead of the seven remaining bits of the original byte
                # Confused? - see vInt at https://lucene.apache.org/core/3_5_0/fileformats.html

                y = int.from_bytes(file_content, byteorder='little', signed=False)
                binstring1 = format(y, "b")
                if len(binstring1) == 8:
                    file_content = file.read(1)
                    y2 = int.from_bytes(file_content, byteorder='little', signed=False)
                    binstring2 = format(y2, "b")
                    y = int(binstring2 + binstring1[1:], 2)

                file_content = file.read(y)
                self.terms.append(file_content.decode())
                file_content = file.read(int(unitsize * dimension))
                q = np.frombuffer(file_content, np.uint8)
                self.bitvectors.append(q)
                file_content = file.read(1)

        self.vectors = [BinaryVectorFactory.generate_vector(args) for args in self.bitvectors]
        self.bitvectors = np.asarray(self.bitvectors, dtype=np.uint8)
        self.dict = dict(zip(self.terms,self.vectors))

    def write_vectors(self, filename):
        """
        Write out binary vector store in Semantic Vectors binary format
        """
        svu.write_packed_binaryvectors(self,filename)

    def get_vector(self,term):
        """
        Return vector representation of term, or None if not found
        """
        return self.dict.get(term)

    def put_vector(self,term, vector):
        """
        Add term and corresponding vector to the store
        """
        self.terms.append(term)
        self.vectors.append(vector)
        #todo replace with append the new bit
        self.bitvectors = [args.bitset for args in self.vectors]
        self.bitvectors = np.asarray(self.bitvectors, dtype=np.uint8)
        self.dict.update({term: vector})

    def normalize_all(self):
        """
        Normalize all vectors in the space (todo - speed up via broadcasting)
        """
        for vector in self.vectors:
            vector.normalize()

    def knn_term(self,term,k,stdev=False):
        """
        Returns k-nearest nieghbors of an incoming term, or None if term not found
        :param term: term to search for
        :param k: number of neigbhors
        :return: list of score/term pairs
        """
        vec = self.dict.get(term)
        if vec is None:
            return None
        return self.knn(vec,k,stdev)


    def knn(self,binary_vector,k,stdev=False):
        """
        Returns k-nearest neighbors of an incoming BinaryVector
        :param: binary_vector (BinaryVector)
        :param: k - number of neighbors
        :return: list of score/term pairs
        """
        if k > len(self.terms):
            k = len(self.terms)
        sims = np.sum(BinaryVector.hd(binary_vector.bitset, self.bitvectors, BinaryVector.lookuptable), axis=1)
        indices = np.argpartition(sims, k)[:k]
        indices = indices[
            np.argsort(sims[indices])]  # to preserve ordering; about 2 times as fast as sort with a lambda fnc
        results = []
        if stdev:
            sims = -zscore(sims)
            for index in indices:
                results.append([sims[index], self.terms[index]])
        else:
            for index in indices:
                results.append([2 * (0.5 - sims[index] / binary_vector.dimension), self.terms[index]])
        return results


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

    def __init__(self, dimension, incoming_vector=None):
        self.dimension = dimension
        if incoming_vector is None:
            self.bitset = np.zeros(int(dimension/8),dtype=np.uint8)
        else:
            self.bitset = incoming_vector
        self.pvr = []
        self.nvr = []
        self.cv  = None
        self.ev  = None

    def voting_record_to_bitset(self):
        """
        Initializes the voting record and matches this to the current bitset. In many cases this
        won't be necessary, as a voting record is only required for superposition, and
        takes up 32x the RAM of the bitset.
        """
        if self.bitset.count(True) > 0:
            self.pvr.append(self.bitset)#np.packbits(self.bitset))
            #self.nvr.append(np.packbits(bitarray('0' * len(self.bitset))))
            dimstring = '>' + str(int(len(self.bitset) * 0.125)) + 'B'
            self.nvr.append(struct.unpack(dimstring, bitarray('0' * len(self.bitset)).tobytes()))

    def set(self, incoming_bitarray):
        """
        Sets the bit vector and voting record to the incoming bitarray (not a copy)
        :param incoming_bitarray: a bitarray.bitarray object
        """
        self.bitset = np.packbits(incoming_bitarray)

    def copy(self):
        """
        :return New BinaryVector with (deep) copies of bit vector and voting record
        """
        new = BinaryVector(self.dimension)
        new.bitset = self.bitset.copy()  # replaces 0 bitset
        if len(self.pvr) > 0:
            new.pvr = self.pvr.copy()  # replaces 0 positive voting_record
            new.nvr = self.nvr.copy()  # replaces 0 negative voting_record

        return new

    def set_random_vector(self):
        """
        Sets the bit vector and voting record to a random vector with an equal number of 1s and 0s
        """
        halfdimension = int32(self.dimension / 2)
        randvec = np.concatenate((np.ones(halfdimension, dtype=int8), np.zeros(halfdimension, dtype=int8)))
        np.random.shuffle(randvec)
        self.bitset = np.packbits(list(randvec.astype(bool)))

    def set_zero_vector(self):
        """
        Sets the bit vector to zero
        """
        self.bitset = np.packbits(bitarray([False] * self.dimension))

    def get_dimension(self):
        """
        :return: dimensionality of the vector
        """
        return self.dimension

    @staticmethod
    def get_vector_type():
        return 'BINARY'

    def is_zero_vector(self):
        return not self.bitset.any()

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

    def addvector(self, cv, vr):
        """
        Basic addition operation - adds a bitset to a voting record with a weight of one
        :param cv:
        :param vr:
        :return:
        """

        for i in range(len(vr)):
            vr[i] = vr[i] ^ cv
            cv = cv & ~vr[i]
        if BinaryVector.getcount(cv) > 0:
            vr.append(cv)

    def addfromfloor(self, cv, vr, rowfloor):
        """
       Faster addition operation for higher weights - adds a bitset to a voting record starting
       at the row corresponding to log2(weight), with higher rows representing higher numbers
       (row n represent the nth lowest order bit of a binary number)
       :param cv:
       :param vr:
       :return:
       """
        for i in range(rowfloor, len(vr)):
            vr[i] = vr[i] ^ cv
            cv = cv & ~vr[i]
        if BinaryVector.getcount(cv) > 0: #cv.count(True) > 0:
            vr.append(cv)

    def reduce(self):
        """
        Eliminate unecessary bits to save space in two steps:
            (1) positive and negative records cancel out
            (2) higher order rows that are identical (both zero at this point) are removed
        """
        """
        :return: 
        """
        for i in range(len(self.pvr)):
            c = self.nvr[i].copy()
            self.nvr[i] = self.nvr[i] & ~self.pvr[i]
            self.pvr[i] = self.pvr[i] & ~c

        for i in range(len(self.pvr) - 1, -1, -1):
            if (self.pvr[i] ^ self.nvr[i]).count() == 0:
                del self.pvr[i]
                del self.nvr[i]
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

    def tally_votes(self):
        """
        Sets self bit vector (bitset) to the outcome of tallying the votes of the voting record
        (more 1s than zeros : 1, with ties split at random)
        """
        if len(self.pvr) == 0:
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


