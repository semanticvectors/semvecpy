"""
Graded vectors are used to represent continuous quantities by interpolating between two endpoints.
This ensures that proximity in terms of cosine similarity corresponds to similar quantities.

Vectors for items in a given position are created by binding the vector for the item with the vector for
the position.

This is used in particular for representing the orthography of written words.

The technique was introduced by Wahle, Cohen, Schvaneveldt, Widdows. For more information see:
Orthogonality and Orthography: Introducing Measured Distance into Semantic Space.
Trevor Cohen, Dominic Widdows, Manuel Wahle.
Proceedings of the Seventh International Conference on Quantum Interaction, Leicester, UK, 2013.
"""
import numpy as np
from . import vector_utils as vu


class GradedVectorFactory:
    """
    GradedVectorFactory creates vectors for proportions by interpolation between between given endpoints.
    """
    def __init__(self, dimension: int, field=np.float):
        self.dimension = dimension
        self.alpha_vec = vu.normalize(vu.create_dense_random_vector(dimension, seed=1, field=field))
        self.omega_vec = vu.normalize(vu.create_dense_random_vector(dimension, seed=2, field=field))

    def get_vector_for_proportion(self, proportion: float):
        """
        Gets the vector interpolated by a given proportion between the vector factory's alpha and omega vectors.
        :param proportion: Usually between 0 and 1, though if outside those ranges the result will be an extrapolation.
        :return: Normalized vector between alpha and omega vectors.
        """
        return vu.normalize(np.multiply(self.alpha_vec, 1 - proportion) + np.multiply(self.omega_vec, proportion))


class OrthographicVectorFactory:
    """
    OrthographicVectorFactory looks after creating and storing and matching orthographic word vectors.

    These vectors represent words as a sum of vectors that bind each character with its relative position in the word.
    """
    def __init__(self, dimension, field=np.float):
        self.dimension = dimension
        self.field = field
        self.gvf = GradedVectorFactory(dimension, field=field)
        self.character_vectors = {}
        self.word_vectors = {}

    def get_word_vector(self, word: str):
        """Gets the vector for the given word, creating and storing it if not already present."""
        if word in self.word_vectors:
            return self.word_vectors[word]
        else:
            output_vector = self.make_and_store_word_vector(word)
            return output_vector

    def make_and_store_word_vector(self, word: str):
        """Makes a vector for the given word, adds it to the word_vectors dictionary, and returns the new vector."""
        output_vector = self.make_word_vector(word)
        self.word_vectors[word] = output_vector
        return output_vector

    def make_word_vector(self, word: str):
        """Makes a vector for the given word and returns the new vector."""
        output_vector = np.zeros(self.dimension, dtype=self.field)
        for pos in range(len(word)):
            letter = word[pos]
            hex_key = int(letter.encode().hex(), 16)
            if hex_key not in self.character_vectors:
                self.character_vectors[hex_key] = vu.create_dense_random_vector(
                    self.dimension, seed=hex_key, field=self.field)
            output_vector += vu.bind(
                self.gvf.get_vector_for_proportion((pos + 0.5) / (len(word))),
                self.character_vectors[hex_key],
                field=self.field)
        return output_vector

    def get_k_nearest_neighbors(self, query_word: str, k: int):
        query_vector = self.make_word_vector(query_word)
        return vu.get_k_neighbors_from_pairs(self.word_vectors.items(), query_vector, k)
