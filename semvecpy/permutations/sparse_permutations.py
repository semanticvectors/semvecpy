"""Experiments in permuting and matching sparse vectors.

There are some overlaps and some differences with dense_permutations.py, we haven't attempted to unify these.
"""
import numpy as np
import random

from typing import Iterable, List, Tuple

from ..vectors import vector_utils as vu
from . import constants as c
from . import dense_permutations as dp


def get_random_sparse_vector(dim: int, entries: int) -> np.array:
    new_vector = np.zeros(dim)
    populated_entries = 0
    while populated_entries < entries:
        position = random.randint(0, dim - 1)
        if new_vector[position] == 0:
            new_vector[position] = random.random()
            populated_entries += 1
    return new_vector


def get_sort_permutation(vector: Iterable[float]) -> List[Tuple[int, float]]:
    """Returns the permutation that sorts the input array descending"""
    indexed_vector = [(i, vector[i]) for i in range(len(vector))]
    sorted_indexed_vector = sorted(indexed_vector, key=lambda x: x[1], reverse=True)
    return list([x[0] for x in sorted_indexed_vector])


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
    vector1 = get_random_sparse_vector(c.DIMENSION, c.SEED_ENTRIES)
    vu.normalize(vector1)
    vector2 = get_random_sparse_vector(c.DIMENSION, c.SEED_ENTRIES)
    vu.normalize(vector2)
    print("Similarity before sorting:", vu.cosine_similarity(vector1, vector2))
    perm_vector1 = dp.permute_vector(get_sort_permutation(vector1), vector1)
    perm_vector2 = dp.permute_vector(get_sort_permutation(vector2), vector2)
    print("Similarity after sorting:", vu.cosine_similarity(perm_vector1, perm_vector2))


if __name__ == '__main__':
    main()

