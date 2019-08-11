import numpy as np

import vectors.vector_utils as vu


class GradedVectorFactory:
    def __init__(self, dimension: int):
        self.dimension = dimension
        self.alpha_vec = vu.normalize(vu.create_dense_random_vector(dimension, seed=1))
        self.omega_vec = vu.normalize(vu.create_dense_random_vector(dimension, seed=2))

    def get_vector_for_proportion(self, proportion: float):
        """
        Gets the vector interpolated by a given proportion between the vector factory's alpha and omega vectors.
        :param proportion: Usually between 0 and 1, though if outside those ranges the result will be an extrapolation.
        :return: Normalized vector between alpha and omega vectors.
        """
        return vu.normalize(np.multiply(self.alpha_vec, 1 - proportion) + np.multiply(self.omega_vec, proportion))


class OrthographicVectorFactory:
    def __init__(self, dimension):
        self.dimension = dimension
        self.gvf = GradedVectorFactory(dimension)
        self.elemental_vectors = {}
        self.word_vectors = {}

    def get_vector(self, word: str):
        if word in self.word_vectors:
            return self.word_vectors[word]

        output_vector = np.zeros(self.dimension, dtype=np.float32)
        for pos in range(len(word)):
            letter = word[pos]
            hex_key = int(letter.encode().hex(), 16)
            if hex_key not in self.elemental_vectors:
                self.elemental_vectors[hex_key] = vu.create_dense_random_vector(self.dimension, seed=hex_key)
            output_vector += vu.circular_convolution(
                self.gvf.get_vector_for_proportion(pos / (len(word) - 1)), self.elemental_vectors[hex_key])

        self.word_vectors[word] = output_vector
        return output_vector
