import numpy as np


def normalize(vector):
    """
    Compute the L2 norm (||v||2) for a vector, v.
    Returns the normalized vector (same dtype).
    Note that spares permutations modifies the vector in place in contrast.
    See dense_permutations_test.py test case difference.
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
