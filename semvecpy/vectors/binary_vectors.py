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
from numpy import int8, int32
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
    def generate_vector(incoming_bitarray):
        binaryvec = BinaryVector(len(incoming_bitarray),incoming_vector=incoming_bitarray)
        return binaryvec


class BinaryVectorStore(object):
    """
    Storage, retrieval and nearest neighbor search of binary vectors
    """
    def __init__(self):
        self.dict = {}
        self.vectors = []
        self.terms = []

    def init_from_lists(self,terms,bitarrays):
        """
        Initializes from lists (e.g. those used by semvec_utils)
        :param terms: list of terms
        :param binary_vectors: list of binary vectors
        """
        self.terms=terms
        self.vectors = [BinaryVectorFactory.generate_vector(args) for args in bitarrays]
        self.dict = dict(zip(self.terms, self.vectors))

    def init_from_file(self,file_name):
        """
        Reads bit vectors from disk (Semantic Vectors binary format) into BinaryVector objects
        Creates a dictionary for retrieval
        :param filename:
        """
        with open(file_name, mode='rb') as file:  # b is important -> binary
            file_content = file.read(1)
            x = file_content
            ct = int.from_bytes(x, byteorder='little', signed=False)
            file_content = file.read(ct)
            header = file_content.decode().split(" ")
            vindex = header.index('-vectortype')
            vectortype = header[vindex + 1]

        if vectortype != 'BINARY':
            print('Can\'t initialize binary vector store from ',vectortype,' vectors.')
            return

        self.terms, incoming_bits = svu.readfile(file_name)
        self.vectors = [BinaryVectorFactory.generate_vector(args) for args in incoming_bits]
        self.dict = dict(zip(self.terms,self.vectors))

    def write_vectors(self, filename):
        """
        Write out binary vector store in Semantic Vectors binary format
        """
        svu.write_bitarray_binaryvectors(self,filename)


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
        return self.dict.update({term: vector})

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

        sims = []
        if k > len(self.terms):
            k = len(self.terms)
        sims = [binary_vector.measure_overlap(args) for args in self.vectors]

        if stdev:
            sims = zscore(sims)
        indices = np.argpartition(sims, -k)[-k:]
        indices = sorted(indices, key=lambda i: sims[i], reverse=True)
        results = []
        for index in indices:
            results.append([sims[index], self.terms[index]])
        return results


class BinaryVector(object):
    """
    BinaryVector objects include both a bit vector (bitset) and a voting record, and
    support the fundamental Vector-symbolic Architecture (see operations of binding and superposition.
    """

    def __init__(self, dimension, incoming_vector=None):
        self.dimension = dimension
        if incoming_vector is None:
            self.bitset = bitarray(dimension*[False])
        else:
            self.bitset = incoming_vector
        self.voting_record = None


    def voting_record_to_bitset(self):
        """
        Initializes the voting record and matches this to the current bitset. In many cases this
        won't be necessary, as a voting record is only required for superposition, and
        takes up 32x the RAM of the bitset.
        """
        self.voting_record = np.zeros(self.dimension)
        if self.bitset.count(True) > 0:
            as_list = np.float(1) * np.array(self.bitset.tolist())
            as_list[as_list == 0] = np.float(-1)
            self.voting_record = as_list

    def set(self, incoming_bitarray):
        """
        Sets the bit vector and voting record to the incoming bitarray (not a copy)
        :param incoming_bitarray: a bitarray.bitarray object
        """
        self.bitset = incoming_bitarray

    def copy(self):
        """
        :return New BinaryVector with (deep) copies of bit vector and voting record
        """
        new = BinaryVector(self.dimension)
        new.bitset = self.bitset.copy()  # replaces 0 bitset
        if self.voting_record is not None:
            new.voting_record = self.voting_record.copy()  # replaces 0 voting_record
        return new

    def set_random_vector(self):
        """
        Sets the bit vector and voting record to a random vector with an equal number of 1s and 0s
        """
        halfdimension = int32(self.dimension / 2)
        randvec = np.concatenate((np.ones(halfdimension, dtype=int8), np.zeros(halfdimension, dtype=int8)))
        np.random.shuffle(randvec)
        self.bitset = bitarray(list(randvec.astype(bool)))

    def set_zero_vector(self):
        """
        Sets the bit vector to zero
        """
        self.bitset = bitarray([False] * self.dimension)

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
        nhd = 1 - (2 / self.dimension) * (self.bitset ^ other.bitset).count(True)
        return np.max(nhd, 0)

    def superpose(self, other, weight):
        """
        Adds the bit vector of the incoming BinaryVector 'other' to the voting record of self
        :param other: another BinaryVector
        """
        other_list = 1 * np.array(other.bitset.tolist())
        other_list[other_list == 0] = -1
        if (self.voting_record is None): #initiliaze voting record for first superposition
            self.voting_record_to_bitset()
        self.voting_record += other_list * weight

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
        if self.voting_record is None: return  # nothing to tally
        s = np.sign(self.voting_record)
        s[s == 0] = np.random.choice(np.array([-1, 1]),
                                     s[s == 0.].shape[0])  # make as many random checks as their are 0s
        s[s == -1] = 0  # set all negatives to 0
        self.bitset = bitarray(list(s.astype(bool)))

    def normalize(self):
        """
        Tallies votes and resets the voting record to None
        """
        self.tally_votes()
        self.voting_record = None


