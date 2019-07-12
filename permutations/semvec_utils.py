"""Some methods to work with pre-existing Semantic Vectors spaces"""

import struct
import copy
import numpy as np
from bitstring import BitArray

def getvector(wordvecs,term):
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
    query_vec = getvector(vectors,query_term)
    return get_k_neighbors(vectors, query_vec, k)

def get_k_neighbors(vectors, query_vec, k):
    """Returns the nearest neighboring terms to query_vec - a real vector"""
    results=[]
    sims = np.matmul(vectors[1], query_vec)
    indices = np.argpartition(sims, -k)[-k:]
    indices = sorted(indices, key=lambda i: sims[i], reverse=True)
    for index in indices:
        label=vectors[0][index]
        results.append(label)
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
    for vector in bwordvectors[1]:
        vec2 = copy.copy(vector)
        vec2 ^= query_vec
        sims.append(-vec2.bin.count("1"))
    indices = np.argpartition(sims, -k)[-k:]
    indices = sorted(indices, key=lambda i: sims[i], reverse=True)
    labels = []
    for index in indices:
        labels.append(bwordvectors[0][index])
    return labels


def readfile(fileName):
    """Read in a Semantic Vectors binary (.bin) file - currently works for real vector, binary vector and permutation stores"""
    words = []
    vectors = []

    with open(fileName, mode='rb') as file:  # b is important -> binary
        fileContent = file.read()

    # determine length of header string (the first byte)
    x = fileContent[0]
    ct = x + 1
    header = fileContent[1:ct].decode().split(" ")
    vindex = header.index('-vectortype')
    vectortype = header[vindex + 1]
    dindex = header.index('-dimension')
    dimension = int(header[dindex + 1])
    unitsize = 4  # bytes per vector dimension
    #print(dimension, " ", vectortype)
    if vectortype == 'REAL':
        dimstring = '>' + str(dimension) + 'f'
    elif vectortype == 'PERMUTATION':
        dimstring = '>' + str(dimension) + 'i'
    elif vectortype == 'BINARY':
        unitsize = .125

    skipcount = 0
    count = 0

    while (ct < len(fileContent)):
        y = int.from_bytes(fileContent[ct:ct + 1], byteorder='little', signed=False)

        # Read Lucene's vInt - if the most significant bit
        # is set, read another byte as significant bits
        # ahead of the seven remaining bits of the original byte
        # Confused? - see vInt at https://lucene.apache.org/core/3_5_0/fileformats.html

        binstring1 = format(y, "b")
        if len(binstring1) == 8:
            y2 = int.from_bytes(fileContent[ct + 1:ct + 2], byteorder='little', signed=False)
            binstring2 = format(y2, "b")
            y = int(binstring2 + binstring1[1:], 2)
            # print('y',y)
            # skip the bit we have just read
            ct = ct + 1
            #print((fileContent[ct + 1:ct + y + 1].decode()))

        words.append(fileContent[ct + 1:ct + y + 1].decode())
        ct = ct + y + 1
        if vectortype == 'BINARY':
            v = int.from_bytes(fileContent[ct:ct + int(unitsize * dimension)], byteorder='little', signed=False)
            binv = format(v, "b")
            toadd = dimension - len(binv)
            binv = str(0) * toadd + binv
            q = BitArray(bin=binv)
        else:
            q = struct.unpack(dimstring, fileContent[ct:ct + int(unitsize * dimension)])

        vectors.append(q)

        ct = ct + int(dimension * unitsize)
        count = count + 1

    return (words, vectors)

