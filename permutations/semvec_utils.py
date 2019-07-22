"""Some methods to work with pre-existing Semantic Vectors spaces"""

import struct
import copy
import numpy as np
from bitarray import bitarray
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
        results.append([sims[index],label])
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
        #.5 - normalized hamming distance
        sims.append( 2*(.5*len(vec2)-vec2.count(True))/ len(vec2))
    indices = np.argpartition(sims, -k)[-k:]
    indices = sorted(indices, key=lambda i: sims[i], reverse=True)
    results = []
    for index in indices:
        results.append([sims[index],bwordvectors[0][index]])
    return results


def readfile(fileName):
    """Read in a Semantic Vectors binary (.bin) file - currently works for real vector, binary vector and permutation stores"""
    words = []
    vectors = []

    with open(fileName, mode='rb') as file:  # b is important -> binary
        fileContent = file.read(1)

        # determine length of header string (the first byte)
        x = fileContent
        ct = int.from_bytes(x, byteorder='little', signed=False)
        fileContent = file.read(ct)
        header = fileContent.decode().split(" ")
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

        fileContent = file.read(1)
        while fileContent:
            # y = int.from_bytes(fileContent[ct:ct + 1], byteorder='little', signed=False)

            # Read Lucene's vInt - if the most significant bit
            # is set, read another byte as significant bits
            # ahead of the seven remaining bits of the original byte
            # Confused? - see vInt at https://lucene.apache.org/core/3_5_0/fileformats.html

            y = int.from_bytes(fileContent, byteorder='little', signed=False)
            binstring1 = format(y, "b")
            if len(binstring1) == 8:
                fileContent = file.read(1)
                y2 = int.from_bytes(fileContent, byteorder='little', signed=False)
                binstring2 = format(y2, "b")
                y = int(binstring2 + binstring1[1:], 2)

            fileContent = file.read(y)
            words.append(fileContent.decode())
            fileContent = file.read(int(unitsize * dimension))

            if vectortype == 'BINARY':
                q=bitarray()
                q.frombytes(fileContent)
            else:
                q = struct.unpack(dimstring, fileContent)

            vectors.append(q)
            fileContent = file.read(1)

    return (words, vectors)

