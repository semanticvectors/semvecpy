import numpy as np


def normalize(vector):
    """
    Returns the normalized vector (same dtype) using the L2 norm (||v||2) for a vector, v.
    """
    return vector / np.sqrt(np.dot(vector, vector))


def cosine_similarity(vector1, vector2):
    """
    Returns the cosine of the angle between vector1 and vector 2
    """
    vector1 = np.array(vector1).astype(np.float64)
    vector2 = np.array(vector2).astype(np.float64)
    norm1 = np.sqrt(np.dot(vector1, vector1))
    norm2 = np.sqrt(np.dot(vector2, vector2))
    return np.dot(vector1, vector2) / (norm1*norm2)


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
