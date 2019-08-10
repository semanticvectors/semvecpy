import numpy as np

import vectors.vector_utils as vu


def circular_convolution(vec1, vec2):
    """
    Returns a vector that is the circular convolution of vec1 and vec2.

    Implementation is one of several suggestions from
    https://stackoverflow.com/questions/35474078/python-1d-array-circular-convolution

    vec1 and vec2: array-like, e.g., list of floats
    """
    return np.real(np.fft.ifft(np.fft.fft(vec1) * np.fft.fft(vec2)))


def create_dense_random_vector(dimension: int, seed=None):
    if seed:
        np.random.seed(seed)
    return np.random.uniform(low=-1, high=1, size=dimension)


class GradedVectorFactory:
    def __init__(self, dimension):
        self.dimension = dimension
        self.alpha_vec = vu.normalize(create_dense_random_vector(dimension, seed=100))
        self.omega_vec = vu.normalize(create_dense_random_vector(dimension, seed=-100))

    def get_vector_for_proportion(self, proportion):
        return np.multiply(self.alpha_vec, 1 - proportion) + np.multiply(self.omega_vec, proportion)
