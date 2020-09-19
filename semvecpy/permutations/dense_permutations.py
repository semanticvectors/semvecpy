"""Experiments in permuting and matching dense vectors.

There are some overlaps and some differences with sparse_permutations.py, we haven't attempted to unify these.
"""
import numpy as np
import random
from semvecpy.permutations import constants as c
from semvecpy.vectors.vector_utils import normalize, cosine_similarity


def get_random_vector(dimension):
    """
    Generate a dense random normal vector of a given dimension.
    Returns a 32bit float vector as a numpy array of shape (100,).
    """
    return np.random.randn(dimension).astype(np.float32)


def get_sort_permutation(vector):
    """
    Returns the permutation that sorts the input array in descending order.
    Permutation vectors are index vectors, being a permutation of the 
    integers between 0 and the dimension of a vector minus one. 
    """
    return np.argsort(vector)[::-1] #[::-1] to get descending order


def get_random_permutation(dimension):
    """
    Returns a random permutation vector for a given dimension.
    Permutation vectors are index vectors, being a permutation of the 
    integers between 0 and the dimension of a vector minus one 
    (equivalent to np.arange(dimension)).
    """
    return np.random.permutation(np.arange(dimension))


def permute_vector(permutation, vector):
    """
    Returns the result of applying the given permutation to the given vector.
    Input permutation and vector should be of the same length.
    Input permutation should be an index vector (e.g. a permutation of
    the integers from 0 to dimension-1 of the vector)
    """
    return vector[permutation]


def inverse_permutation(permutation):
    """
    Returns the inverse of a given permutation.
    Input permutation should be an index vector (e.g. a permutation of
    the integers from 0 to dimension-1 of the vector). Note that for this
    inverse to work, each number in the range should occur exactly once
    (i.e. is a true index vector).
    """
    return np.argsort(permutation)


def swap_permutation(permutation, numswaps):
    """
    Swap numswaps pairs of integers in the permutation
    Input permutation should be an index vector (e.g. a permutation of
    the integers from 0 to dimension-1 of the vector).
    """
    newperm = permutation.copy()
    for x in range(numswaps):
        a, b = random.sample(range(len(permutation)), 2)
        newperm[a],newperm[b] = newperm[b],newperm[a]
    return newperm

def permutation_to_matrix(permutation):
    """
    Returns a matrix version of a permutation, as a linear transformation.
    Permutation should be a 1 dimensional index vector.
    Returned matrix should be multiplied on the right to behave as permutation. TODO: Decide if we want it this way.
    """
    perm_matrix = np.zeros((permutation.shape[0], permutation.shape[0]), dtype=np.float32)
    perm_matrix[permutation, np.arange(permutation.shape[0])] = 1
    return perm_matrix


def main():
    vector1 = get_random_vector(c.DIMENSION)
    normalized_vector1 = normalize(vector1)
    vector2 = get_random_vector(c.DIMENSION)
    normalized_vector2 = normalize(vector2)
    print("")
    print("Normalized or not, values should be the same (give or take a rounding error).")
    print("For dense vectors, they should generally be around 0.5 similarity. Higher dimensions",
          "will be more consistent.")
    print("Similarity before sorting, no normalization: %.4f" % cosine_similarity(vector1, vector2))
    print("Similarity before sorting, normalized: %.4f" % cosine_similarity(normalized_vector1, normalized_vector2))
    print("\n")

    perm_vector1 = permute_vector(get_sort_permutation(vector1), vector1)
    perm_vector2 = permute_vector(get_sort_permutation(vector2), vector2)
    norm_perm_vector1 = permute_vector(get_sort_permutation(normalized_vector1), normalized_vector1)
    norm_perm_vector2 = permute_vector(get_sort_permutation(normalized_vector2), normalized_vector2)
    print("For dense vectors with a sort permutation, they are likely to be more similar after permuting.")
    print("Similarity after sorting, no normalization: %.4f" % cosine_similarity(perm_vector1, perm_vector2))
    print("Similarity after sorting, normalized: %.4f" % cosine_similarity(norm_perm_vector1, norm_perm_vector2))
    print("\n")

    randompermvec = get_random_permutation(c.DIMENSION)
    randompermvec2 = get_random_permutation(c.DIMENSION)
    rperm_vector1 = permute_vector(randompermvec, vector1)
    rperm_vector2 = permute_vector(randompermvec, vector2)
    rnorm_perm_vector1 = permute_vector(randompermvec, normalized_vector1)
    rnorm_perm_vector2 = permute_vector(randompermvec, normalized_vector2)
    rperm2_vector2 = permute_vector(randompermvec2, vector2)
    rnorm_perm2_vector2 = permute_vector(randompermvec2, normalized_vector2)
    print("For dense vectors with random permutations, they should still be around 0.5 similarity.")
    print("With identical permutation, they should have the same similarity as before permutation:")
    print("Similarity after sorting, no normalization: %.4f" % cosine_similarity(rperm_vector1, rperm_vector2))
    print("Similarity after sorting, normalized: %.4f" % cosine_similarity(rnorm_perm_vector1, rnorm_perm_vector2))
    print("")
    print("And with two different permutations, we should get differing values still trending to 0.5 similarity:")
    print("Similarity after sorting, no normalization: %.4f" % cosine_similarity(rperm_vector1, rperm2_vector2))
    print("Similarity after sorting, normalized: %.4f" % cosine_similarity(rnorm_perm_vector1, rnorm_perm2_vector2))
    print("")


if __name__ == '__main__':
    main()

