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
from bitarray import bitarray


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
        binaryvec = BinaryVector(len(incoming_bitarray))
        binaryvec.set(incoming_bitarray)
        return binaryvec


class BinaryVector(object):
    """
    BinaryVector objects include both a bit vector (bitset) and a voting record, and
    support the fundamental Vector-symbolic Architecture (see operations of binding and superposition.
    """

    def __init__(self, dimension):
        self.dimension = dimension
        self.bitset = None
        self.voting_record = None
        self.set_zero_vector()

    def set(self, incoming_bitarray):
        """
        Sets the bit vector and voting record to the incoming bitarray (not a copy)
        :param incoming_bitarray: a bitarray.bitarray object
        """
        self.bitset = incoming_bitarray
        self.voting_record = np.zeros(self.dimension)
        as_list = 1 * np.array(self.bitset.tolist())
        as_list[as_list == 0] = -1
        self.voting_record = as_list

    def copy(self):
        """
        :return New BinaryVector with (deep) copies of bit vector and voting record
        """
        new = BinaryVector(self.dimension)
        new.bitset = self.bitset.copy()  # replaces 0 bitset
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
        randvec[randvec == 0] = -1
        self.voting_record = randvec

    def set_zero_vector(self):
        """
        Sets the bit vector and voting record to zero
        """
        self.bitset = bitarray([False] * self.dimension)
        self.voting_record = np.zeros(self.dimension)

    def get_dimension(self):
        return self.dimension

    @staticmethod
    def get_vector_type():
        return 'binary'

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
        if np.sum(np.abs(self.voting_record)) == 0: return  # this shortcut only occurs after a normalization
        s = np.sign(self.voting_record)
        s[s == 0] = np.random.choice(np.array([-1, 1]),
                                     s[s == 0.].shape[0])  # make as many random checks as their are 0s
        s[s == -1] = 0  # set all negatives to 0
        self.bitset = bitarray(list(s.astype(bool)))

    def normalize(self):
        """
        Tallies votes and resets the voting record to reflect the new bitset
        """
        self.tally_votes()
        self.voting_record = np.sign(self.voting_record)


def main():
    vector1 = BinaryVector(2048)
    vector1.set_random_vector()
    vector1c = vector1.copy()
    vector2 = BinaryVector(2048)

    for i in range(10):
        vector2.set_random_vector()
        vector1.superpose(vector2, 1)
        if i < 100:
            vector1.tally_votes()
            print(i, vector1.bitset.count(True), vector1.measure_overlap(vector1c))


if __name__ == '__main__':
    main()
