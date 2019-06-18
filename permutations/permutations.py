"""Experiments in permuting and matching sparse vectors."""
import math
import numpy as np
import random

from typing import Dict, Iterable, List, Tuple

DIMENSION = 10
ENTRIES = 2


def get_random_vector(dim: int, entries: int) -> np.array:
    new_vector = np.zeros(dim)
    populated_entries = 0
    while populated_entries < entries:
        position = random.randint(0, dim - 1)
        if new_vector[position] == 0:
            new_vector[position] = random.random()
            populated_entries += 1
    return new_vector


def normalize(vector: Iterable[float]) -> List[float]:
    norm = math.sqrt(sum([x * x for x in vector]))
    for i in range(len(vector)):
        vector[i] /= norm


def get_sort_permutation(vector: Iterable[float]) -> List[Tuple[int, float]]:
    """Returns the permutation that sorts the input array descending"""
    indexed_vector = [(i, vector[i]) for i in range(len(vector))]
    sorted_indexed_vector = sorted(indexed_vector, key=lambda x: x[1], reverse=True)
    return list([x[0] for x in sorted_indexed_vector])


def permute_vector(perm: List[int], vector: Iterable[float]) -> np.array:
    """Returns the result of applying the given permutation to the given vector

    Input perm and vector must be of the same length.
    """
    permuted_vector = np.zeros(len(vector))
    for i in range(len(perm)):
        permuted_vector[perm[i]] = vector[i]
    return permuted_vector


def inverse_permutation(perm: List[int]):
    """Returns the inverse permutation.

    Input perm must be a permutation of the integers from 0 to len(perm) - 1"""
    inverse = [0] * len(perm)
    for index in range(len(perm)):
        inverse[perm[index]] = index
    return inverse


def permutation_to_matrix(perm: List[int]) -> np.array:
    """Returns a matrix version of a permutation, as a linear transformation"""
    perm_matrix = np.zeros([len(perm), len(perm)])
    for i in range(len(perm)):
        perm_matrix[perm[i], i] = 1
    return perm_matrix


def main():
    vector1 = get_random_vector(DIMENSION, ENTRIES)
    normalize(vector1)
    vector2 = get_random_vector(DIMENSION, ENTRIES)
    normalize(vector2)
    print("Similarity before sorting:", sum(vector1 * vector2))
    perm_vector1 = permute_vector(get_sort_permutation(vector1), vector1)
    perm_vector2 = permute_vector(get_sort_permutation(vector2), vector2)
    print("Similarity after sorting:", sum(perm_vector1 * perm_vector2))


if __name__ == '__main__':
    main()

