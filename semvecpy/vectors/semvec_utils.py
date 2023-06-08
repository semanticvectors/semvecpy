"""Some methods to work with pre-existing Semantic Vectors spaces"""

import struct
import os
from tqdm import tqdm
import copy
from typing import List
import numpy as np
from bitarray import bitarray
import sys

def getvector(wordvecs, term):
    """
    Retrieve the vector for a term
    Parameters are a pair of lists, wordvecs[0] - the terms, wordvecs[1] - the vectors
    """
    if term in wordvecs[0]:
        index = wordvecs[0].index(term)
        return wordvecs[1][index]
    else:
        return None


def get_k_vec_neighbors(vectors, query_term, k):
    """Returns the nearest neighboring terms to query_term - a term."""
    query_vec = getvector(vectors, query_term)
    return get_k_neighbors(vectors, query_vec, k)


def get_k_neighbors(vectors, query_vec, k):
    """Returns the nearest neighboring terms to query_vec - a real vector"""
    results = []
    sims = np.matmul(vectors[1], query_vec)
    indices = np.argpartition(sims, -k)[-k:]
    indices = sorted(indices, key=lambda i: sims[i], reverse=True)
    for index in indices:
        label = vectors[0][index]
        results.append([sims[index], label])
    return results


def get_k_bvec_neighbors(bwordvectors, query_term, k):
    """Returns the nearest neighboring terms (binary vector reps) to query_term - a term"""
    if query_term in bwordvectors[0]:
        query_index = bwordvectors[0].index(query_term)
        query_vec = bwordvectors[1][query_index]
        return get_k_b_neighbors(bwordvectors, query_vec, k)
    else:
        return None


def get_k_b_neighbors(bwordvectors, query_vec, k):
    """Returns the nearest neighboring to terms to query_vec - a binary vector."""
    sims = []
    if k > len(bwordvectors[0]):
        k = len(bwordvectors[0])
    for vector in bwordvectors[1]:
        nnhd = measure_overlap(query_vec, vector)
        sims.append(nnhd)
    indices = np.argpartition(sims, -k)[-k:]
    indices = sorted(indices, key=lambda i: sims[i], reverse=True)
    results = []
    for index in indices:
        results.append([sims[index], bwordvectors[0][index]])
    return results


def search(term: str, search_vectors, elemental_vectors=None, semantic_vectors=None,
           predicate_vectors=None, count=20, search_type="single_term"):
    """
    Search for terms that have representations that are similar to the given term's vector.
    :param term: search term (word or expression)
    :param search_vectors: the vector file to search for similar terms
    :param elemental_vectors:
    :param semantic_vectors:
    :param predicate_vectors:
    :param count: number of results to return.
    :param search_type: currently supported: boundproduct or single_term. If single_term is specified, it is assumed that the term comes from search_vectors and search_vectors will be searched for other terms that are similar to the given term. If boundproduct is specified, the expression will be resolved using the supplied vectors and the search_vectors will be searched for the resulting vector.
    :return: Top {count} most similar terms from search_vectors.
    """
    if search_type != "single_term" and search_type != "boundproduct":
        raise NotImplementedError()

    if search_type == "single_term":
        return get_k_bvec_neighbors(search_vectors, term, count)
    else:
        if search_type == "boundproduct" and (elemental_vectors is None
                                              or semantic_vectors is None
                                              or predicate_vectors is None):
            raise ValueError("All three vector files must be provided if search type is boundproduct")

        v = get_bound_product_query_vector_from_string(term, elemental_vectors=elemental_vectors,
                                                       semantic_vectors=semantic_vectors,
                                                       predicate_vectors=predicate_vectors)
        return get_k_b_neighbors(search_vectors, v, count)


def compare_terms_batch(terms, elemental_vectors, semantic_vectors, predicate_vectors) -> List[float]:
    """
    Compares the terms in the specified list of term comparisons.
    :param elemental_vectors:
    :param semantic_vectors:
    :param predicate_vectors:
    :param terms: List of terms to compare. Each line should contain a single string containing exactly one pipe (|) character. Bound products (*) are allowed.
    :return:
    """
    similarities = list()
    for term in terms:
        if term.count("|") != 1:
            raise ValueError("Input must contain exactly one | character per line")
        term1, term2 = tuple(term.split("|"))
        similarities.append(compare_terms(term1, term2,
                                          elemental_vectors=elemental_vectors,
                                          semantic_vectors=semantic_vectors,
                                          predicate_vectors=predicate_vectors))
    return similarities


def compare_terms(term1: str, term2: str, elemental_vectors, semantic_vectors, predicate_vectors,
                  search_type: str = "boundproduct") -> float:
    """
    Look up the vector representations for the two given terms and determine the similarity between them.
    :param elemental_vectors:
    :param semantic_vectors:
    :param predicate_vectors:
    :param term1: First word or term for comparison.
    :param term2: Second word or term for comparison
    :param search_type: Search type. Currently, only boundproduct is supported.
    :return: Similarity (normalized hamming distance) between the vectors for the two terms.
    """
    if search_type != "boundproduct":
        raise NotImplementedError()

    return measure_overlap(
        get_bound_product_query_vector_from_string(term1, elemental_vectors=elemental_vectors,
                                                   semantic_vectors=semantic_vectors,
                                                   predicate_vectors=predicate_vectors),
        get_bound_product_query_vector_from_string(term2, elemental_vectors=elemental_vectors,
                                                   semantic_vectors=semantic_vectors,
                                                   predicate_vectors=predicate_vectors))


def measure_overlap(vector1, vector2, binary: bool = True) -> float:
    """
    Returns the similarity (0.5-normalized hamming distance) between the two given vectors.
    :param vector1: A vector
    :param vector2: A vector
    :param binary: True if the vectors to be compared are binary vectors.
    :return: Similarity. Higher number means more similarity. 1.0 means the vectors are identical. Can be negative.
    """
    if not binary:
        raise NotImplementedError()

    vec2 = copy.copy(vector2)
    vec2 ^= vector1
    # .5 - normalized hamming distance
    nnhd = 2 * (.5 * len(vec2) - vec2.count(True)) / len(vec2)
    return nnhd


def get_vector_for_token(token: str, elemental_vectors, semantic_vectors, predicate_vectors) -> bitarray:
    """
    :param elemental_vectors:
    :param semantic_vectors:
    :param predicate_vectors:
    :param token: A string token such as P(side_effect) or S(drug). More complex expressions are not currently supported
    :return:
    """
    if token[0] == "P":
        vectors = predicate_vectors
    elif token[0] == "E" or token[0] == "C":
        vectors = elemental_vectors
    elif token[0] == "S":
        vectors = semantic_vectors
    else:
        raise MalformedQueryError("Vector set identifier must be P, E, or S (was", token[0], ")")

    token = token[2:-1]
    if token not in vectors[0]:
        raise TermNotFoundError(token)

    query_index = vectors[0].index(token)
    return vectors[1][query_index]


def get_bound_product_query_vector_from_string(query: str, elemental_vectors, semantic_vectors,
                                               predicate_vectors) -> bitarray:
    """
    :param elemental_vectors:
    :param semantic_vectors:
    :param predicate_vectors:
    :param query: calculate the bound product of the terms to be bound in this query. E.g. a query of E(south_africa)*S(pretoria) would result in the elemental vector for South Africa and the semantic vector for Pretoria being looked up; then their bound product is returned.
    :return: Vector for the specified query term.
    """
    if "|" in query:
        raise NotImplementedError()

    if "+" in query:
        raise NotImplementedError()

    result = None
    tokens = query.split("*")
    for token in tokens:
        v = copy.copy(get_vector_for_token(token, elemental_vectors=elemental_vectors,
                                           semantic_vectors=semantic_vectors,
                                           predicate_vectors=predicate_vectors))
        if result is None:
            result = v
        else:
            result ^= v
    return result


class TermNotFoundError(ValueError):
    def __init__(self, term: str, *args: object) -> None:
        super().__init__("Term not found:", term)


class MalformedQueryError(ValueError):
    def __init__(self, query: str, *args: object) -> None:
        super().__init__("Not a valid query:", query)


def readfile(file_name):
    """
    Read in a Semantic Vectors binary (.bin) file - currently works for real vector, binary vector and permutation stores
    """
    words = []
    vectors = []

    filestats = os.stat(file_name)
    filesize = int(filestats.st_size)

    with open(file_name, mode='rb') as file:  # b is important -> binary
        file_content = file.read(1)

        # determine length of header string (the first byte)
        x = file_content
        ct = int.from_bytes(x, byteorder='little', signed=False)
        file_content = file.read(ct)
        header = file_content.decode().split(" ")
        vindex = header.index('-vectortype')
        vectortype = header[vindex + 1]
        dindex = header.index('-dimension')
        dimension = int(header[dindex + 1])
        unitsize = 4  # bytes per vector dimension
        print(dimension, " ", vectortype)
        if vectortype == 'REAL':
            dimstring = '>' + str(dimension) + 'f'
        elif vectortype == 'PERMUTATION':
            dimstring = '>' + str(dimension) + 'i'
        elif vectortype == 'BINARY':
            unitsize = .125

        skipcount = 0
        count = 0
        pbar = tqdm(total=filesize)
        file_content = file.read(1)

        while file_content:
            # y = int.from_bytes(file_content[ct:ct + 1], byteorder='little', signed=False)

            # Read Lucene's vInt - if the most significant bit
            # is set, read another byte as significant bits
            # ahead of the seven remaining bits of the original byte
            # Confused? - see vInt at https://lucene.apache.org/core/3_5_0/fileformats.html

            y = int.from_bytes(file_content, byteorder='little', signed=False)
            binstring1 = format(y, "b")
            if len(binstring1) == 8:
                file_content = file.read(1)
                y2 = int.from_bytes(file_content, byteorder='little', signed=False)
                binstring2 = format(y2, "b")
                y = int(binstring2 + binstring1[1:], 2)

            file_content = file.read(y)
            pbar.update(y)
            words.append(file_content.decode())
            file_content = file.read(int(unitsize * dimension))
            pbar.update(int(unitsize * dimension))
            if vectortype == 'BINARY':
                q = bitarray()
                q.frombytes(file_content)
            else:
                q = np.asarray(struct.unpack(dimstring, file_content))

            vectors.append(q)
            file_content = file.read(1)
            pbar.update(1)
    pbar.close()
    return (words, vectors)

def get_vint(i):
    """
    Utility function to replicate Lucene's variable length integer format
    """
    b = format(i,"b")
    if (len(b) < 8):
        b = str(0)*(8-len(b))+b
        return bitarray(b)
    else:
        b2 = format(i%128,"b")
        b2 = str(1)+str(0)*(7-len(b2))+b2
        b3 = format(i//128,"b")
        b3 = str(0)*(8-len(b3))+b3
        return bitarray(b2+b3)

def write_realvectors(vecstore, filename):
    """
        Write out real vector store in Semantic Vectors binary format
    """
    with open(filename, mode='wb') as file:  # b is important -> binary
        x = '-vectortype REAL -dimension '+str(np.asarray(vecstore.vectors).shape[1])
        file.write((len(x)).to_bytes(1,byteorder='little', signed=False))
        file.write(x.encode('utf-8'))
        for word in vecstore.dict.keys():
            vint = get_vint(len(word.encode('utf-8')))
            file.write(vint)
            file.write(word.encode('utf-8'))
            floats = vecstore.get_vector(word).vector
            s = struct.pack('>'+str(len(floats))+'f', *floats)
            file.write(s)

def write_bitarray_binaryvectors(vecstore, filename):
    """
        Write out binary vector store in Semantic Vectors binary format
    """
    with open(filename, mode='wb') as file:  # b is important -> binary
        x = '-vectortype BINARY -dimension ' + str(vecstore.vectors[0].dimension)
        file.write((len(x)).to_bytes(1, byteorder='little', signed=False))
        file.write(x.encode('utf-8'))
        for word in vecstore.dict.keys():
            vint = get_vint(len(word.encode('utf-8')))
            file.write(vint)
            file.write(word.encode('utf-8'))
            bins = vecstore.get_vector(word).bitset
            file.write(bins)


def write_packed_binaryvectors(vecstore, filename):
    """
        Write out packed binary vector store in Semantic Vectors binary format
    """
    with open(filename, mode='wb') as file:  # b is important -> binary
        x = '-vectortype BINARY -dimension ' + str(vecstore.vectors[0].dimension)
        file.write((len(x)).to_bytes(1, byteorder='little', signed=False))
        file.write(x.encode('utf-8'))
        for word in vecstore.dict.keys():
            vint = get_vint(len(word.encode('utf-8')))
            file.write(vint)
            file.write(word.encode('utf-8'))
            bins = vecstore.get_vector(word).bitset
            file.write(bins)


def pathfinder(q, r, dish, cosines=True):
    """
        Returns 'pruned' Pathfinder network preserving shortest paths within constraints:
        :param q: max length of paths to consider
        :param r: determines minkowski distance
        :param dish: pairwise similarities between nodes, scale 0-1
        :return: the pruned matrix
    """
    n = len(dish)

    q = np.minimum(q, n - 1)
    dis = np.zeros([n,n],dtype=np.float)
    mindis = np.zeros([n,n],dtype=np.float)
    changedatq = np.zeros([n,n],dtype=np.int)

    for row in range(n):
        for col in range(n):
            if cosines:
                dis[row][col] = 1 - dish[row][col]
                mindis[row][col] = 1 - dish[row][col]
            else:
                dis[row][col] = dish[row][col]
                mindis[row][col] = dish[row][col]


    if q > n - 2:
        topass = 1
        #Floyd's algorithm for minimum distance
        for ind in range(n):
         for row in range(n):
            for col in range(n):
                indirect = minkowski(mindis[row][ind], mindis[ind][col], r)

        #if (indirect < minDis[row][col]) {
                if (mindis[row][col] - indirect) > 1e-10:
                    mindis[row][col] = indirect
                    changedatq[row][col] = topass

    else:
         # Dijkstra's algorithm for minimum distance

        topass = 1
        changed = True
        while changed and topass < q:

            topass += 1
            changed = False
            m = copy.copy(mindis)
            for ind in range(n):
                for row in range(n):
                    for col in range(n):
                        indirect1 = minkowski(dis[row][ind], m[ind][col], r)
                        indirect2 = minkowski(m[row][ind], dis[ind][col], r)
                        indirect = np.minimum(indirect1, indirect2)
                        #if (indirect < mindis[row][col]) - replaced with line below, correction courtesy Roger Schvaneveldt
                        if mindis[row][col] - indirect > 1e-10:
                            mindis[row][col] = indirect
                            changedatq[row][col] = topass
                            changed = True

    pruned = np.zeros([n,n],dtype=np.float)

    for row in range(n):
        for col in range(n):
            if mindis[row][col] == dis[row][col]:
                if cosines:
                    pruned[row][col] = 1-dis[row][col]
                else:
                    pruned[row][col] =  dis[row][col]

    return pruned


def minkowski (x,y,r):
    """
        Returns minkowski distance between scalars x and y, parameterized by r
        :param x:
        :param y:
        :param r:
        :return: the distance
    """

    if r == np.inf or np.minimum(x, y) == 0:
        return np.maximum(x, y)
    elif r==1:
        return x+y
    else:
        return np.power((np.power(x, r) + np.power(y, r)), (1 / r))


