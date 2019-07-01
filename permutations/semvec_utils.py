"""Some methods to work with pre-existing Semantic Vectors spaces"""

import struct

import numpy as np

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
    """Returns the nearest neighboring terms to query_vec."""
    query_vec = getvector(vectors,query_term)
    return get_k_neighbors(vectors, query_vec, k)

def get_k_neighbors(vectors, query_vec, k):
    results=[]
    sims = np.matmul(vectors[1], query_vec)
    indices = np.argpartition(sims, -k)[-k:]
    indices = sorted(indices, key=lambda i: sims[i], reverse=True)
    for index in indices:
        label=vectors[0][index]
        results.append(label)
    return results

def readfile(fileName):
    """
        Read a Semantic Vectors Binary file returning a pair of lists, wordvecs[0] - the terms, wordvecs[1] - the vectors
        Currently only REAL and PERMUTATION vector stores supported
    """
    words=[]
    vectors=[]

    with open(fileName, mode='rb') as file: # b is important -> binary
        fileContent = file.read()

    #determine length of header string (the first byte)
    x=fileContent[0]
    print(x)
    ct= x + 1
    header = fileContent[1:ct].decode().split(" ")
    vindex = header.index('-vectortype')
    vectortype = header[vindex+1]
    dindex = header.index('-dimension')
    dimension = int(header[dindex+1])
    print(dimension," ",vectortype)
    dimstring = '>'+str(dimension)+'f'
    pimstring = '>'+str(dimension)+'i'

    skipcount = 0
    count = 0

    while (ct < len(fileContent)):
        y=int.from_bytes(fileContent[ct:ct+1],byteorder='little',signed=False)
        try:
            #Read Lucene's vInt - if the most significant bit
            #is set, read another byte as significant bits
            #ahead of the seven remaining bits of the original byte
            #Confused? - see vInt at https://lucene.apache.org/core/3_5_0/fileformats.html

            binstring1=format(y,"b")
            if len(binstring1) == 8:
                 y2=int.from_bytes(fileContent[ct+1:ct+2],byteorder='little',signed=False)
                 binstring2=format(y2,"b")
                 y=int(binstring2+binstring1[1:],2)
                 print('y',y)
                 #skip the bit we have just read
                 ct = ct + 1
                 print((fileContent[ct+1:ct+y+1].decode()))

            words.append(fileContent[ct+1:ct+y+1].decode())
            ct=ct+y+1
            if vectortype == 'PERMUTATION':
                q=struct.unpack(pimstring,fileContent[ct:ct+(4*dimension)])
            else:
                q=struct.unpack(dimstring,fileContent[ct:ct+(4*dimension)])
            vectors.append(q)
        except:
            skipcount = skipcount + 1
        ct=ct+dimension*4
        count = count +1

    print('skipped '+str(skipcount))
    return(words,vectors)


