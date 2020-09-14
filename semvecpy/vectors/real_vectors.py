"""
Real vectors provide the basis for semantic vectors (word embeddings), and
faciliate a representational approach developed by Tony Plate known as
Holographic Reduced Representations, with
the following key components:
(1) Randomly generated dense real vectors
(2) A *superposition* operator: vector addition
(3) A *binding* operator: circular convolution (via fft)

For further details of the Holographic Reduced Representations, see:

- Plate TA. Holographic Reduced Representation: Distributed Representation for Cognitive Structures.
Stanfpord, CA.: CSLI Publications; 2003.

For further discussion of Vector Symbolic Architectures in general, see:

- Gayler RW. Vector Symbolic Architectures answer Jackendoff’s challenges for cognitive neuroscience.
In: In Peter Slezak (Ed), ICCS/ASCS International Conference on Cognitive Science. Sydney, Australia. University of New South Wales.; 2004. p. 133–8.

- Kanerva P. Hyperdimensional computing: An introduction to computing in distributed representation
with high-dimensional random vectors. Cognitive Computation. 2009;1(2):139–159.

"""

import numpy as np
from scipy.stats import zscore
import copy
from . import semvec_utils as svu
from . import vector_utils as vu


class RealVectorFactory:
    """
    RealVectorFactory creates three sorts of RealVector objects:
    (1) Zero vectors
    (2) Random vectors (real vectors with random values)
    (3) Vectors with preset values (e.g. read from disk)
    """

    @staticmethod
    def generate_vector(incoming_vector):
        realvec = RealVector(len(incoming_vector), incoming_vector)
        return realvec

    @staticmethod
    def generate_random_vector(dimension):
        randvec = np.random.uniform(low=-.5/dimension, high=.5/dimension, size=dimension)
        return RealVectorFactory.generate_vector(randvec)

    @staticmethod
    def generate_zero_vector(dimension):
        zerovec = np.zeros(dimension)
        return RealVectorFactory.generate_vector(zerovec)




class RealVectorStore(object):
    """
    Storage, retrieval and nearest neighbor search of real vectors
    """
    def __init__(self):
        self.dict = {}
        self.vectors = []
        self.real_vectors = []
        self.terms = []

    def init_from_lists(self,terms,vectors):
        """
        Initializes from lists (e.g. those used by semvec_utils)
        :param terms: list of terms
        :param real_vectors.py: list of real vectors
        """
        self.terms = terms
        self.vectors = vectors
        self.real_vectors = [RealVectorFactory.generate_vector(args) for args in vectors]
        self.dict = dict(zip(self.terms, self.real_vectors))

    def init_from_file(self,file_name):
        """
        Reads bit vectors from disk (Semantic Vectors binary format) into BinaryVector objects
        Creates a dictionary for retrieval
        :param file_name:
        """
        with open(file_name, mode='rb') as file:  # b is important -> binary
            file_content = file.read(1)
            x = file_content
            ct = int.from_bytes(x, byteorder='little', signed=False)
            file_content = file.read(ct)
            header = file_content.decode().split(" ")
            vindex = header.index('-vectortype')
            vectortype = header[vindex + 1]

        if vectortype != 'REAL':
            print('Can\'t initialize real vector store from ',vectortype,' vectors.')
            return

        #read in vectors and wrap in RealVectors
        incoming_terms, incoming_vectors = svu.readfile(file_name)
        self.init_from_lists(incoming_terms,incoming_vectors)

    def write_vectors(self, filename):
        """
        Write out real vector store in Semantic Vectors binary format
        """
        svu.write_realvectors(self,filename)

    def get_vector(self,term):
        """
        Return vector representation of term, or None if not found
        """
        return self.dict.get(term)

    def put_vector(self, term, vector):
        """
        Add term and corresponding vector to the store
        """
        self.terms.append(term)
        self.vectors.append(vector.vector)
        self.real_vectors.append(vector)
        return self.dict.update({term: vector})

    def normalize_all(self):
        """
        Normalize all vectors in the space (todo - speed up via broadcasting)
        """
        #for i, vector in enumerate(self.real_vectors):
        #    self.real_vectors[i] /= np.linalg.norm(vector)
        self.vectors /= np.linalg.norm(self.vectors, axis=1).reshape(-1,1)
        for i, vector in enumerate(self.real_vectors):
            vector.set(self.vectors[i])

        
    def knn_term(self,term,k, stdev=False):
        """
        Returns k-nearest nieghbors of an incoming term, or None if term not found
        :param term: term to search for
        :param k: number of neigbhors
        :return: list of score/term pairs
        """
        real_vec = self.dict.get(term)
        if real_vec is None:
            return None
        return self.knn(real_vec,k, stdev=stdev)

    def knn(self,query_vec,k, stdev=False):
        """
        Returns k-nearest neighbors of an incoming RealVector
        :param: query_vec (RealVector)
        :param: k - number of neighbors
        :return: list of score/term pairs
        """

        sims = []
        if k > len(self.terms):
            k = len(self.terms)
        sims = np.matmul(self.vectors, query_vec.vector)
        if stdev:
            sims = zscore(sims)
        indices = np.argpartition(sims, -k)[-k:]
        indices = sorted(indices, key=lambda i: sims[i], reverse=True)
        results = []
        for index in indices:
            results.append([sims[index], self.terms[index]])
        return results


class RealVector(object):
    """
    RealVector objects include both a numpy array of floats (a vector), and
    support the fundamental Vector-symbolic Architecture operations of binding and superposition.
    """

    #governs whether to use approximate or exact inverse of circular convolution
    exact_convolution_inverse = False;

    def __init__(self, dimension, incoming_vector=None):
        self.dimension = dimension
        if incoming_vector is None:
            self.vector = np.zeros(dimension)
        else:
            self.vector = np.asarray(incoming_vector)

    def set(self, incoming_vector):
        """
        Sets the vector and to the incoming numpy array (not a copy)
        :param incoming_vector: a numpy array of floats
        """
        self.vector = incoming_vector

    def copy(self):
        """
        :return New RealVector with (deep) copies of vector
        """
        new = RealVector(self.dimension)
        new.vector = copy.copy(self.vector)
        return new

    def set_random_vector(self):
        """
        Sets the vector to a dense random vector
        """
        self.vector = vu.create_dense_random_vector(dimension)

    def set_zero_vector(self):
        """
        Sets the vector to zero
        """
        self.vector = np.zeros(self.dimension, dtype = float)

    def get_dimension(self):
        """
        :return: dimensionality of the vector
        """
        return self.dimension

    @staticmethod
    def get_vector_type():
        return 'REAL'

    def is_zero_vector(self):
        return (np.linalg.norm(self.vector) == 0)


    def measure_overlap(self, other):
        """
        # Scalar product - cosine distance would be an alternative, but we may wish
        # to leave normalization out of the basic functionality and impose it as needed
        # (it's probably better to this once off en masse than on a per-comparison basis anyhow)
        :param other: another RealVector
        :return the scalar product
        """
        return np.dot(self.vector, other.vector)

    def superpose(self, other, weight):
        """
        Adds the incoming RealVector 'other' to the numpy array of self
        :param other: another RealVector
        """
        self.vector += np.multiply(weight,other.vector)

    def bind(self, other):
        """
        Binds the vector of the incoming RealVector 'other' to the vector of self
        :param other: another RealVector
        """
        self.vector = vu.circular_convolution(self.vector, other.vector)

    def release(self, other):
        """
        The inverse of the bind operator, which with real vectors is circular correlation
        :param other: another RealVector
        """
        if RealVector.exact_convolution_inverse:
            self.vector = vu.exact_circular_correlation(self.vector,other.vector)
        else:
            self.vector = vu.circular_convolution(self.vector,other.involution())

    def normalize(self):
        """
        Tallies votes and resets the voting record to None
        """
        self.vector /= np.linalg.norm(self.vector)

    def involution(self):
        """
        Gets involution of vector (as numpy array) for approximate circular correlation where
        e.g. involution [0,1,2,3] = [0,3,2,1]
        :return: involution: a numpy array containing the involution of the RealVector's numpy array
        """
        involution_index = [0]
        involution_index.extend(range((self.dimension - 1), 0, -1))
        return self.vector[involution_index]
