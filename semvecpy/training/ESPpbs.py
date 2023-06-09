from tqdm import tqdm
import argparse
import mmap
import traceback
from bitarray import bitarray
import time
from semvecpy.vectors import binary_packedbitvectors as bv

from semvecpy.vectors import semvec_utils as svu
import numpy as np
from multiprocessing import shared_memory
from functools import partial
from multiprocessing import Process, Pool, Lock
import re
import multiprocessing
import random


# convert integer to number of 1 bits in its packed binary representation (for hamming distance)
lookuptable = np.asarray([0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4, 1, 2, 2, 3, 2, 3,
                          3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4,
                          3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 1, 2,
                          2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5,
                          3, 4, 4, 5, 4, 5, 5, 6, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5,
                          5, 6, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 1, 2, 2, 3,
                          2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4,
                          4, 5, 4, 5, 5, 6, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
                          3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 2, 3, 3, 4, 3, 4,
                          4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6,
                          5, 6, 6, 7, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 4, 5,
                          5, 6, 5, 6, 6, 7, 5, 6, 6, 7, 6, 7, 7, 8], dtype=np.uint8)

#discard and refresh voting record
def normalize_all(sign,links,vrs):
    numvecs = len(sign)
    links[0:numvecs] = range(numvecs)
    links[numvecs:] = -1
    vrs[:] = vrs[:]^vrs[:]
    print('buffer overflow - normalizing', np.sum(vrs[numvecs:]), np.sum(links[numvecs:]))


# batched matrix edition

#compress all to be added to a single subject
def count_sumvotes(incoming,weights):
    insum = np.sum([np.unpackbits(x)*weights[i] for i,x in enumerate(incoming)],axis=0) / np.sum(weights)
    r = np.random.rand(insum.shape[0])
    #insum[insum == 0.5] = np.random.choice(np.array([0, 1]),
    #
    #insum[insum == 0.5].shape[0])  # make as many random checks as their are 0s

    #probabilistic sampling according to votes
    insum = (r < insum)
    return np.packbits(np.asarray(insum,dtype=np.uint8))


def msmaddfromfloor(inbits, vrs, links, sign, indexes, rowfloors):
    """
   Faster addition operation for higher weights - adds a bitset to a voting record starting
   at the row corresponding to log2(weight), with higher rows representing higher numbers
   (row n represent the nth lowest order bit of a binary number)
   :param cv:
   :param vr:
   :return:
   """
    traces = indexes.copy()
    previndexes = indexes.copy()
    # ensure prerequisite voting records exist
    seekCount = 0

    for i in range(np.max(rowfloors) + 1):
        seekCount += 1
        if seekCount > 10:
            print('seek', indexes)

        # print(i,'min_traces',np.min(traces))

        active_indices = indexes >= 0
        previndexes[active_indices] = indexes[active_indices]
        traces[(rowfloors >= i) & (active_indices)] = previndexes[(rowfloors >= i) & (active_indices)]
        # indexes[indexes >= 0] = links[indexes[indexes >= 0]]
        indexes[active_indices] = links[indexes[active_indices]]
        # indexes[(indexes >= 0) & (indexes == previndexes)] = -1
        indexes[(active_indices) & (indexes == previndexes)] = -1
        indexes[(rowfloors >= i) & (active_indices) & (indexes == previndexes)] = -1
        startnew = int(np.max(links) + 1)
        if startnew >= len(links):
            print('overflow')
        else:
            lock.acquire()
            toupdate = np.unique(np.asarray(previndexes)[(indexes == -1) & (rowfloors >= i)])
            nuindexes = np.asarray(range(startnew, startnew + len(toupdate)), dtype=int)
            links[toupdate] = nuindexes
            nudict = dict(zip(toupdate, nuindexes))
            if len(nuindexes) > 0:
                nuadditions = np.asarray(
                    [nudict[p] for p in np.asarray(previndexes)[(indexes == -1) & (rowfloors >= i)]], dtype=int)
                indexes[(rowfloors >= i) & (indexes == -1)] = nuadditions  # nuindexes
            lock.release()
    indexes = traces

    msmadd(inbits, vrs, links, sign, indexes)


def msmsub(inbits, vrs, links, sign, indexes):
    origindexes = indexes
    previndexes = indexes.copy()
    cvs = inbits.copy()

    traces = []
    while np.max(indexes) >= 0:
        active_indices = indexes >= 0
        toupdate = indexes[active_indices]
        traces.append(indexes)  # [active_indices])
        vrs[toupdate] = vrs[toupdate] ^ cvs[active_indices]
        cvs[active_indices] = cvs[active_indices] & vrs[toupdate]
        previndexes[active_indices] = indexes[active_indices]
        indexes = links[indexes]
        indexes[indexes == previndexes] = -1

    counts = np.asarray([bv.BinaryVector.getcount(x) for x in cvs])
    if (np.sum(counts) > 0):
        sign[origindexes[counts > 0]] = sign[origindexes[counts > 0]] ^ cvs[counts > 0]  # switch signs as required
        # toupdate = np.asarray(traces)  # [t for t in traces])
        # toupdate = toupdate[toupdate >= 0]

        for t in traces:
            vrs[t[counts > 0]] = vrs[t[counts > 0]] ^ cvs[counts > 0]
        msmadd(cvs.copy()[counts > 0], vrs, links, sign, origindexes[counts > 0])


def msmsuperposefromfloor(inbits, vrs, links, sign, indexes, floors):
    pinbits = np.invert(inbits ^ sign[indexes])  # agrees with sign
    ninbits = ~pinbits  # disagrees with sign
    msmaddfromfloor(pinbits.copy(), vrs, links, sign, indexes.copy(), floors.copy())
    msmsubfromfloor(ninbits, vrs, links, sign, indexes, floors)


def msmsuperpose(tinbits, vrs, links, sign, indexes, weights):
    inbits = tinbits.copy()  # protect the incoming bits
    inbits[weights < 0] = np.invert(inbits[weights < 0].copy())
    weights = np.abs(weights)

    while np.max(weights) > 0:  # 0 < np.max(rowfloors) < len(vrs) and np.max(weights) > 0:
        inbits = inbits[weights > 0]
        if indexes.shape != weights.shape:
            print('---->', indexes.shape, weights.shape)
        indexes = indexes[weights > 0]
        weights = weights[weights > 0]

        if len(weights) == 0:
            return

        rowfloors = np.array((np.floor(np.log2(weights))), dtype=int)
        rowfloors = rowfloors[weights > 0]
        msmsuperposefromfloor(inbits, vrs, links, sign, indexes, rowfloors)

        weights = weights - np.asarray(np.power(2, rowfloors), dtype=int)

        # rowfloors = np.array((np.floor(np.log2(weights[weights > 0]))),dtype=int)

    # while np.max(weights) > 0:  # incrementally add the rest
    #    msmsuperposefromfloor(inbits[weights > 0],vrs,links,sign,indexes[weights > 0],np.zeros_like(indexes[weights > 0]))
    #    weights -= 1


def msmadd(inbits, vrs, links, sign, indexes):
    previndexes = indexes.copy()
    cvs = inbits.copy()

    seekCount = 0
    # todo impose indexes >= 0 constraint throughout
    while np.max(indexes) >= 0:
        active_indices = indexes >= 0
        seekCount += 1
        if seekCount > 50:
            print('seek', indexes)

        toupdate = indexes[active_indices]
        vrs[toupdate] = vrs[toupdate] ^ cvs[active_indices]
        cvs[active_indices] = cvs[active_indices] & ~vrs[toupdate]
        # print('v0 =',countvotes(0,links,vrec,signs),'\nv1 =',countvotes(1,links,vrec,signs))
        # print(np.unpackbits(cvs[active_indices]))
        previndexes[active_indices] = indexes[active_indices].copy()

        indexes[active_indices] = links[indexes[active_indices]]
        indexes[(active_indices) & (indexes == previndexes)] = -1
        # print(indexes)

    counts = np.asarray([bv.BinaryVector.getcount(x) for x in cvs])

    startnew = int(np.max(links) + 1)
    if startnew >= len(links):
        print('overflow')
    else:
        lock.acquire()
        toupdate = np.unique(np.asarray(previndexes)[counts > 0])
        nuindexes = np.asarray(range(startnew, startnew + len(toupdate)), dtype=int)
        nudict = dict(zip(toupdate, nuindexes))
        nuadditions = np.asarray([nudict[p] for p in np.asarray(previndexes)[counts > 0]], dtype=int)
        links[toupdate] = nuindexes
        # print('extending from msmadd',nuindexes)
        # vrs[nuadditions] = cvs[counts > 0]
        lock.release()
        for i, na in enumerate(nuadditions):
            # print(i,na)
            smadd(cvs[counts > 0][i], vrs, links, sign, na) #check this......
            # vrs[i] = cvs[counts > 0][i]
        # msmadd(cvs[counts > 0], vrs, links, sign, nuadditions)
        # vrs[nuindexes] = cvs[counts > 0]



def msmsubfromfloor(inbits, vrs, links, sign, indexes, rowfloors):
    # print('v0 =',countvotes(0,links,vrs,sign),'\tv1 =',countvotes(1,links,vrs,sign))
    # print('rowfloors',rowfloors)
    if np.min(rowfloors) == 0:
        msmsub(inbits[rowfloors == 0], vrs, links, sign, indexes[rowfloors == 0])

    if np.max(rowfloors) == 0:
        return

    indexes = indexes[rowfloors > 0]
    inbits = inbits[rowfloors > 0]
    rowfloors = rowfloors[rowfloors > 0]
    origindexes = indexes.copy()
    previndexes = indexes.copy()
    cvs = inbits.copy()
    traces = indexes
    traces2 = []

    # ensure prerequisite voting records exist
    # adding should already have happened during positive phase
    for i in range(np.max(rowfloors) + 1):
        active_indices = indexes >= 0
        previndexes[active_indices] = indexes[active_indices]
        # traces2.append(indexes[(rowfloors >= i)].copy())
        # TODO: work out how to only update those examples where rowfloor exceeds amount added so far
        icopy = indexes.copy()
        icopy[(rowfloors < i)] = -1
        traces2.append(icopy)
        traces[(rowfloors >= i) & (active_indices)] = previndexes[(rowfloors >= i) & (active_indices)]
        indexes = links[indexes]
        indexes[(rowfloors >= i) & (indexes == previndexes)] = -1
        # startnew = np.int(np.max(links)+1)
        # toupdate = np.asarray(previndexes)[(indexes == -1) & (rowfloors >= i)]
        # nuindexes = np.asarray(range(startnew,startnew+len(toupdate)),dtype=int)
        # links[toupdate] = nuindexes
        # indexes[(rowfloors >= i) & (indexes == -1)] = nuindexes

    indexes = traces
    traces2 = traces2[:-1]
    rowcount = i

    while np.max(indexes) >= 0:
        traces2.append(indexes.copy())  # [indexes >= 0].copy())
        toupdate = indexes[indexes >= 0]
        vrs[toupdate] = vrs[toupdate] ^ cvs[indexes >= 0]
        cvs[indexes >= 0] = cvs[indexes >= 0] & vrs[toupdate]
        # print('\n-v0 =',countvotes(0,links,vrs,sign),'\nv1 =',countvotes(1,links,vrs,sign))

        counts = np.asarray([bv.BinaryVector.getcount(x) for x in cvs])

        previndexes = indexes
        # rowfloors[indexes >= 0] -= 1
        indexes[indexes >= 0] = links[indexes[indexes >= 0]]
        indexes[indexes == previndexes] = -1

        # rowfloors = np.maximum(0,rowfloors)
    counts = np.asarray([bv.BinaryVector.getcount(x) for x in cvs])
    # print('v0 =',countvotes(0,links,vrs,sign),'\tv1 =',countvotes(1,links,vrs,sign))

    # print('traces2',traces2)

    if (np.sum(counts) > 0):
        # print('counts',counts)

        prevt = traces2[-1]
        for t in traces2:
            #print(t)
            try:
                t[t == prevt] = -1
                prevt = t
                tcounts = counts[t >= 0]
                tc = t[t >= 0]
                tcvs = cvs[t >= 0]
                vrs[tc[tcounts > 0]] = vrs[tc[tcounts > 0]] ^ tcvs[tcounts > 0]
            except:
                traceback.print_exc()
            # print('v0 =',countvotes(0,links,vrs,sign),'\tv1 =',countvotes(1,links,vrs,sign))
        print('flipped')
        sign[origindexes[counts > 0]] = sign[origindexes[counts > 0]] ^ cvs[counts > 0]  # switch signs as required

        # msmaddfromfloor(cvs.copy()[counts > 0],vrs,links,sign,origindexes[counts > 0],rowfloors[counts > 0])

        # msmaddfromfloor(cvs.copy()[counts > 0],vrs,links,sign,origindexes[counts > 0],rowfloors[counts > 0])

        msmadd(cvs.copy()[counts > 0], vrs, links, sign, origindexes[counts > 0])


# matrix edition
def smadd(inbit, vrs, links, sign, index):
    previndex = index
    origindex = index
    cv = inbit.copy()
    while index >= 0:
        vrs[index] = vrs[index] ^ cv
        cv = cv & ~vrs[index]
        previndex = index
        index = links[index]
        if index == previndex:
            index = -1

    if bv.BinaryVector.getcount(cv) > 0:
        lock.acquire()
        newindex = int(np.max(links) + 1)
        if newindex > len(vrs): #- overflow condition
            normalize_all(sign,links,vrs)
            newindex = int(np.max(links) + 1)
        links[previndex] = int(newindex)

        vrs[newindex] = cv
        lock.release()


def smaddfromfloor(inbit, vrs, links, sign, index, rowfloor):
    """
   Faster addition operation for higher weights - adds a bitset to a voting record starting
   at the row corresponding to log2(weight), with higher rows representing higher numbers
   (row n represent the nth lowest order bit of a binary number)
   :param cv:
   :param vr:
   :return:
   """
    previndex = index
    origindex = index
    cv = inbit.copy()
    trace = []

    if rowfloor == 0:
        smadd(inbit, vrs, links, sign, index)
        return

    # ensure prerequisite voting records exist
    for i in range(rowfloor + 1):
        trace.append(index)
        previndex = index
        if index > len(vrs): #- overflow condition
            normalize_all(sign,links,vrs)
            index = int(np.max(links) + 1)

        index = links[index]
        if (index == -1) or (index == previndex):
            newindex = int(np.max(links) + 1)
            if newindex > len(vrs):  # - overflow condition
                normalize_all(sign, links, vrs)
                newindex = int(np.max(links) + 1)
            links[previndex] = newindex
            index = newindex

    index = trace[-1]
    smadd(inbit, vrs, links, sign, index)


def smsub(inbit, vrs, links, sign, index):
    origindex = index
    previndex = index
    cv = inbit.copy()

    trace = []
    while index >= 0:
        trace.append(index)
        vrs[index] = vrs[index] ^ cv
        cv = cv & vrs[index]
        previndex = index
        index = links[index]
        if index == previndex:
            index = -1

    if np.sum(cv) > 0:
        sign[origindex] = sign[origindex] ^ cv  # switch signs as required
        # need to invert the bits that have flipped sign
        for t in trace:
            vrs[t] = vrs[t] ^ cv
        smadd(cv.copy(), vrs, links, sign, origindex)


def smsubfromfloor(inbit, vrs, links, sign, index, rowfloor):
    origindex = index
    previndex = index
    cv = inbit.copy()
    trace = []
    if rowfloor == 0:
        smsub(inbit, vrs, links, sign, index)
        return

    # ensure prerequisite voting records exist
    for i in range(rowfloor + 1):
        trace.append(index)
        previndex = index
        index = links[index]
        if (index == -1) or (index == previndex):
            newindex = int(np.max(links) + 1)
            links[previndex] = newindex
            index = newindex

    index = trace[-1]

    while index >= 0:
        vrs[index] = vrs[index] ^ cv
        cv = cv & vrs[index]
        previndex = index
        index = links[index]
        if index == previndex:
            index = -1
        else:
            trace.append(index)

    if np.sum(cv) > 0:
        sign[origindex] = sign[origindex] ^ cv  # switch signs as required
        # need to invert the bits that have flipped sign
        for t in trace:
            vrs[t] = vrs[t] ^ cv
        smadd(cv.copy(), vrs, links, sign, origindex)


def smsuperposefromfloor(inbit, vrs, links, sign, index, floor):
    pinbit = np.invert(inbit ^ sign[index])  # agrees with sign
    ninbit = ~pinbit  # disagrees with sign
    smaddfromfloor(pinbit, vrs, links, sign, index, floor)
    smsubfromfloor(ninbit, vrs, links, sign, index, floor)


# def smsuperpose(inbit,vrs,links,sign,index):
#    pinbit = np.invert(inbit ^ sign[index]) #agrees with sign
#    ninbit = ~pinbit #disagrees with sign
#    smadd(pinbit,vrs,links,sign,index)
#    smsub(ninbit,vrs,links,sign,index)

def smsuperpose(tinbit, vrs, links, sign, index, weight):
    inbit = tinbit.copy()  # protect the incoming bit
    if weight == 0:
        smsuperposefromfloor(inbit, vrs, links, sign, index, 0)
        return
    if weight < 0:
        inbit = np.invert(inbit.copy())
        weight = np.abs(weight)

    rowfloor = int(np.floor(np.log2(weight)))
    while 0 < rowfloor < len(vrs) and weight > 0:
        weight = weight - int(np.power(2, rowfloor))
        smsuperposefromfloor(inbit, vrs, links, sign, index, rowfloor)
        if weight > 0:
            rowfloor = int(np.floor(np.log2(weight)))

    for q in range(weight):  # incrementally add the rest
        smsuperposefromfloor(inbit, vrs, links, sign, index, 0)



def get_num_lines(file_path):
    fp = open(file_path, "r+")
    buf = mmap.mmap(fp.fileno(), 0)
    lines = 0
    while buf.readline():
        lines += 1
    return lines

def process_file(infile):
    counts = {}
    pcounts = {}
    with open(infile) as file:
        for line in tqdm(file, total=get_num_lines(infile)):
            spo = line.replace('\n','').strip().split("\t")
            sub = spo[0].lower().replace(' ','_')
            pred = spo[1].upper().replace(' ','_')
            obj = spo[2].lower().replace(' ','_')

            if sub not in counts:
                counts[sub] = 0
            if obj not in counts:
                counts[obj] = 0
            if pred not in pcounts:
                pcounts[pred] = 0
            counts[sub] += 1
            pcounts[pred] += 1
            counts[obj] += 1
    return counts,pcounts

def nnhd(v1,v2):
    #nnhd = np.maximum(0, 1 - 2*np.count_nonzero(np.unpackbits(v1 ^ v2))/(len(v1)*8))
    nnhd = np.count_nonzero(np.unpackbits(v1^v2))
    nnhd = 1 - 2 * (nnhd / len(np.unpackbits(v1)))
    nnhd = np.maximum(nnhd,0)
    return nnhd

def get_weight(v1, v2, alpha, label):
    weight = label - nnhd(v1, v2)
    return int(weight * alpha)


def countvotes(index, links, pvr, sign):
    counter = np.zeros(len(pvr[0]) * 8, dtype=int)
    previndex = index
    origindex = index
    bitcount = 0

    while index != -1:
        up = np.unpackbits(pvr[index])

        # nup = np.unpackbits(np.invert(pvr[index]))
        counter += np.power(2, bitcount) * up
        # counter -= np.power(2,bitcount)*nup
        bitcount = bitcount + 1
        index = links[index]
        if index == previndex:
            index = -1
        previndex = index

    signe = np.unpackbits(sign[origindex])
    signe = np.asarray(signe, dtype=int)
    signe[signe == 0] = -1
    return counter * signe

def process_lines(shm_names,shm_sizes,shm_types, rdict, lines):
    vecs_shm = shared_memory.SharedMemory(name=shm_names[0])
    links_shm = shared_memory.SharedMemory(name=shm_names[1])
    signs_shm = shared_memory.SharedMemory(name=shm_names[2])
    vrec = np.ndarray(shm_sizes[0], dtype=shm_types[0], buffer=vecs_shm.buf)
    links = np.ndarray(shm_sizes[1], dtype=shm_types[1], buffer=links_shm.buf)
    signs = np.ndarray(shm_sizes[2], dtype=shm_types[2], buffer=signs_shm.buf)

    batch = np.asarray(lines)

    #for b in lines:
    #    print(rdict[b[0]],rdict[b[1]],rdict[b[2]],b[3],b[4])

    c1 = np.asarray(batch[:, 0], dtype=int)
    c2 = np.asarray(batch[:, 1], dtype=int)
    c3 = np.asarray(batch[:, 2], dtype=int)
    c4 = np.asarray(batch[:, 3], dtype=float)  # alpha
    c5 = np.asarray(batch[:, 4], dtype=float)  # labels
    #labels = c5.copy() # preserve to divide batches

    #sims = np.asarray([np.count_nonzero(np.unpackbits(x)) for x in (signs[c1] ^ signs[c2] ^ signs[c3])])

    sims = np.sum(lookuptable[signs[c1] ^ signs[c2] ^ signs[c3]],axis=1)
    sims = 1 - 2 * sims / (8 * signs[c1].shape[1])
    sims = np.maximum(sims, 0)
    c5 = c5 - sims
    c4 = c4 * c5
    c4 = np.array(100*c4,dtype=int)
    nulines = np.column_stack((c1, c2, c3, c4))#.tolist()


    subs = np.asarray(nulines[:,0],dtype=int)
    preds = np.asarray(nulines[:,1],dtype=int)
    objs = np.asarray(nulines[:,2],dtype=int)
    weights = np.asarray(nulines[:,3],dtype=int)
    evecs = signs[objs].copy()
    svecs = signs[subs].copy()
    pvecs = signs[preds].copy()

    hasweight = ~(weights == 0)
    #ignore zero weights
    subs = subs[hasweight]
    objs = objs[hasweight]
    oneway = evecs[hasweight]
    otherway = svecs[hasweight]
    pvecs = pvecs[hasweight]
    weights = weights[hasweight]

    nulines = nulines[hasweight]
    preds = preds[hasweight]
    #for b in nulines.tolist():
    #    print(rdict[b[0]],rdict[b[1]],rdict[b[2]],b[3])

    oneway = oneway ^ pvecs
    otherway = otherway ^ pvecs

    #msmsuperpose(oneway, vrec, links, signs, subs, weights)
    #msmsuperpose(otherway, vrec, links, signs, objs, weights)


    #avoid duplicate entries by summing values for subject / object (preserved code - not used at present)
#def subgrps():
    for i,sumgroup in enumerate([np.asarray(weights > 0), np.asarray(weights < 0)]):
        #print(i,np.sum(sumgroup))
        subset = set(subs)
        unsubs = []
        unvecs = []
        unweights = []

        for sub in subset:
            toadd = np.asarray(subs==sub)
            toadd = toadd & sumgroup


            #print(rdict[sub], np.sum(toadd))

            if np.sum(toadd) > 1:
                #print(-1*i,rdict[sub],[rdict[x] for x in preds[toadd]],[rdict[x] for x in objs[toadd]])
                unvecs.append(count_sumvotes(oneway[toadd],weights[toadd]))
                unweights.append(int(np.average(weights[toadd])))
                unsubs.append(sub)

            elif np.sum(toadd) == 1:
                #print(-1*i,rdict[sub],[rdict[x] for x in objs[toadd]],[rdict[x] for x in objs[toadd]])
                unvecs.append(oneway[toadd][0])
                unweights.append(weights[toadd][0])
                unsubs.append(sub)


        objset = set(objs)
        unobjs = []
        unovecs = []
        unoweights = []

        for obj in objset:
            toadd = np.asarray(objs == obj)
            toadd = toadd & sumgroup
            if np.sum(toadd) > 1:
                #print(-1 * i, rdict[sub], [rdict[x] for x in preds[toadd]], [rdict[x] for x in objs[toadd]])
                unovecs.append(count_sumvotes(otherway[toadd], weights[toadd]))
                unoweights.append(int(np.average(weights[toadd])))
                unobjs.append(obj)

            elif np.sum(toadd) == 1:
                #print(-1 * i, rdict[sub], [rdict[x] for x in preds[toadd]], [rdict[x] for x in objs[toadd]])
                unovecs.append(otherway[toadd][0])
                unoweights.append(weights[toadd][0])
                unobjs.append(obj)

        #print('\nu1',objs[0],countvotes(unobjs[0],links,vrec,signs),sumgroup)
        #print(unoweights[0],'\ni',np.unpackbits(unovecs[0]))
        #print('--->', np.asarray(unsubs).shape, np.asarray(unweights).shape)
        #print('--->',np.asarray(unobjs).shape,np.asarray(unoweights).shape)
        #x = signs[unsubs[0]]
        if len(unsubs) > 0:
            q = signs[unsubs[0]]
            r = unvecs[0]
            pre = np.sum(lookuptable[q^r])
            pre = 1 - 2 * pre / (8 * q.shape[0])
            pre = np.maximum(pre, 0)

            prex =  countvotes(unsubs[0],links,vrec,signs)

            try:
                msmsuperpose(np.asarray(unvecs,dtype=int), vrec, links, signs, np.asarray(unsubs,dtype=int), np.asarray(unweights,dtype=int))
            except:
                print(np.asarray(unweights,dtype=int).shape)
                print(np.asarray(unsubs,dtype=int).shape)
                traceback.print_exc()
            q = signs[unsubs[0]]
            r = unvecs[0]
            post = np.sum(lookuptable[q^r])
            post = 1 - 2 * post / (8 * q.shape[0])
            post = np.maximum(post, 0)
            addee = np.asarray(np.unpackbits(unvecs[0])[:3],dtype=int)
            addee[addee==0] = -1
            print('before',prex[:3],pre,'--->',unweights[0],np.unpackbits(unvecs[0])[:3], "--->",post,'after ', countvotes(unsubs[0], links, vrec, signs)[:3],'ideally',prex[:3]+addee*unweights[0])

        if len(unobjs) > 0:
            msmsuperpose(np.asarray(unovecs,dtype=int), vrec, links, signs, np.asarray(unobjs,dtype=int), np.asarray(unoweights,dtype=int))

        #print('\nu2', unobjs[0], countvotes(unobjs[0], links, vrec, signs),sumgroup)

        #print(subs[0])

    #print('\nb',countvotes(subs[0],links,vrec,signs))
    #print(weights[0])
    #print('\nic',np.unpackbits(evecs[0]^pvecs[0]))


    #for y in range(len(weights)):
    #    smsuperpose(evecs[y] ^ pvecs[y], vrec, links, signs, subs[hasweight][y], weights[y])
    #    smsuperpose(svecs[y] ^ pvecs[y], vrec, links, signs, objs[hasweight][y], weights[y])

    #msmsuperpose(evecs ^ pvecs, vrec, links, signs, subs[hasweight], weights)
    #msmsuperpose(svecs ^ pvecs, vrec, links, signs, objs[hasweight], weights)
    #print(vrec[subs[0]])
    #print('\na', countvotes(subs[0], links, vrec, signs))
    #print('------------------------------')





#else:
        #print('\nskipped',sub,pred,obj)



def process_line(shm_names,shm_sizes,shm_types, line):
    sub = int(line[0])
    pred = int(line[1])
    obj = int(line[2])
    weight = int(line[3])
    #label = line[4]

    vecs_shm = shared_memory.SharedMemory(name=shm_names[0])
    links_shm = shared_memory.SharedMemory(name=shm_names[1])
    signs_shm = shared_memory.SharedMemory(name=shm_names[2])
    vrec = np.ndarray(shm_sizes[0], dtype=shm_types[0], buffer=vecs_shm.buf)
    signs = np.ndarray(shm_sizes[2], dtype=shm_types[2], buffer=signs_shm.buf)
    links = np.ndarray(shm_sizes[1], dtype=shm_types[1], buffer=links_shm.buf)

    evec = signs[obj].copy()
    svec = signs[sub].copy()
    pvec = signs[pred].copy()

    #w1 = get_weight(evec ^ pvec, signs[sub], alpha,label)
    #w2 = get_weight(svec ^ pvec, signs[obj], alpha,label)
    smsuperpose(evec ^ pvec, vrec, links, signs, sub, weight)
    smsuperpose(svec ^ pvec, vrec, links, signs, obj, weight)


#else:
        #print('\nskipped',sub,pred,obj)

def get_triples(inline):
    line, alpha, cdict, ssprob, tlist, nsfrq, numsamples = inline

    batch = []
    spo = line.replace('\n', '').strip().split('\t')
    sub = spo[0].lower().replace(' ', '_')
    pred = spo[1].upper().replace(' ', '_')
    obj = spo[2].lower().replace(' ', '_')

    # both subject and object must make it past subsampling
    # we'll uses the most stringently subsampled of the two to set the bar
    tmax = -1
    if sub in ssprob:
        tmax = ssprob[sub]
    if obj in ssprob:
        tmax = np.maximum(ssprob[obj], tmax)


    if tmax < 0 or np.random.rand() > tmax:
        if sub in cdict and obj in cdict and pred in cdict:
            batch.append([cdict[sub], cdict[pred], cdict[obj + "-OUT"], alpha, 1])
            batch.append([cdict[obj], cdict[pred + "-INV"], cdict[sub + "-OUT"], alpha, 1])
            negsamples = np.random.choice(tlist, numsamples, replace=True, p=nsfrq)

            for negsamp in negsamples:
                batch.append([cdict[sub], cdict[pred], cdict[negsamp + "-OUT"], alpha, 0])
                batch.append([cdict[negsamp], cdict[pred + "-INV"], cdict[sub + "-OUT"], alpha, 0])

    return batch

def init_pool_processes(the_lock):
    """
    Initialize each process with global variable lock.
    https://stackoverflow.com/questions/69907453/lock-objects-should-only-be-shared-between-processes-through-inheritance
    """
    global lock
    lock = the_lock

def train_file(infile,cdict,tlist, shm_names,shm_sizes,shm_types,numsamples,nsfrq,ssprob,epochs, batchsize,rdict):

    initial_alpha = 0.25
    alpha = 0.25
    min_alpha = 0.001


    signs_shm = shared_memory.SharedMemory(name=shm_names[2])
    signs = np.ndarray(shm_sizes[2], dtype=shm_types[2], buffer=signs_shm.buf)

    batch = []
    alphas = []

    tlines = get_num_lines(infile)



    #numpools = multiprocessing.cpu_count()-1
    #numpools = 3
    #pool = Pool(numpools)
    numpools = 1
    lock = multiprocessing.Lock()
    pool = Pool(numpools,initializer=init_pool_processes, initargs=(lock,))



    func = partial(process_lines, shm_names, shm_sizes, shm_types, rdict)

    print('training')
    for epoch in range(epochs):
        print('epoch',epoch,'of',epochs)
        with open(infile, 'r') as file:
            for i,line in enumerate(tqdm(file, total=tlines)):
                alpha = np.maximum(initial_alpha - (initial_alpha - min_alpha) * i / tlines, min_alpha)
                batch += get_triples([line,alpha,cdict,ssprob,tlist,nsfrq, numsamples])
                #print(len(batch), len(batch) % batchsize, tlines-i)
                if ((len(batch) > 0) and len(batch) % (batchsize) == 0) or tlines-i <= 1:

                    #nubatch = np.array_split(nubatch, multiprocessing.cpu_count()*4)
                    #print('\nprocessing', len(batch), 'predications')
                    #for i in tqdm(range(epochs)):
                    #random.shuffle(nubatch)
                    #for _ in tqdm(pool.imap_unordered(func, nubatch), total=len(nubatch)):
                    #   pass
                    #batch = np.asarray(batch)

                    #with Pool(numpools) as pool:
                    #    nubatch = np.array_split(batch,numpools)
                    #    drs = pool.imap_unordered(func, nubatch)#,chunksize = len(batch)//numpools)
                    #    pool.close()
                    #    pool.join()
                    #total = len(nubatch)
                    for _ in tqdm(pool.imap_unordered(func, [batch]), total=len(batch),position=0, leave=True):
                        pass
                    #for a in nubatch:
                    #    for b in a:
                    #        print(rdict[b[0]],rdict[b[1]],rdict[b[2]],b[3],b[4])


                    #pool.close()
                    #pool.join()
                    batch = []
    pool.close()
    pool.join()

def knn(vectors,terms,k, query_vec):
    """
    Returns k-nearest neighbors of an incoming RealVector
    :param: query_vec (RealVector)
    :param: k - number of neighbors
    :return: list of score/term pairs
    """

    sims = []
    if k > len(terms):
        k = len(terms)
    sims = np.matmul(vectors, query_vec)
    #if stdev:
    #    sims = zscore(sims)
    indices = np.argpartition(sims, -k)[-k:]
    indices = sorted(indices, key=lambda i: sims[i], reverse=True)
    results = []
    for index in indices:
        results.append([sims[index], terms[index]])
    return results



def main(args):
    # defaults
    filename = args.filename
    minfrequency = int(args.minfrequency) if args.minfrequency else 50
    maxfrequency = int(args.maxfrequency) if args.maxfrequency else 1000000
    dim = int(args.dim) if args.dim else 128
    numsamples = int(args.numsamples) if args.numsamples else 1
    ssthresh = float(args.ssthresh) if args.ssthresh else 0.001
    bufsize = int(args.bufsize) if args.bufsize else 8
    epochs = int(args.epochs) if args.epochs else 1
    batchsize = int(args.batchsize) if args.batchsize else 1000

    counts, pcounts = process_file(filename)
    interms  = []
    inpreds  = []
    outterms = []

    ssprob = {}
    subcount = 0

    # Total worddcount includes words that don't meet the frequency threshold (but not stopwords)
    total_wordcount = sum(counts.values())

    for x in list(counts.keys()):
        if counts[x] >= minfrequency and counts[x] <= maxfrequency:
            interms.append(x)
            outterms.append(x+'-OUT')
            f = counts[x] / total_wordcount
            if f > ssthresh:
                ssprob.update({x: 1 - np.sqrt(
                    ssthresh / f)})  # use the more straightforward formula from the paper - at least two variants of this exist
            subcount += 1
        else:
            print('skipping',x, counts[x])
            del counts[x]

    for x in list(pcounts.keys()):
        if pcounts[x] >= minfrequency:
            inpreds.append(x)
            inpreds.append(x+'-INV')
        else:
            print('skipping',x,pcounts[x])
            del pcounts[x]


    nsfrq = np.asarray([counts[i] / total_wordcount for i in counts.keys()]) ** 0.75
    nsfrq /= np.linalg.norm(nsfrq, ord=1)

    numterms = len(interms)
    numpreds = len(inpreds)
    print(numterms,'terms meet the minimum frequency threshold of',minfrequency,'and the maximum frequency of',maxfrequency)
    print(numpreds/2,'preds meet the minimum frequency threshold of',minfrequency)

    allterms = interms + outterms + inpreds
    tlist = np.asarray(interms)
    numvecs = len(allterms)

    vrec = np.asarray([np.packbits(np.zeros(dim, dtype=bool))] * bufsize * (numvecs))
    links = np.asarray(-1 * np.ones(bufsize * numvecs), dtype=int)
    signs = []
    for i in range(numvecs):
        signs.append(bv.BinaryVectorFactory.generate_random_vector(dim).bitset)

    signs = np.asarray(signs,dtype=np.uint8)
    links[0:numvecs] = range(numvecs)
    shm = shared_memory.SharedMemory(create=True, size=(vrec.nbytes))
    shm1 = shared_memory.SharedMemory(create=True, size=(links.nbytes))
    shm2 = shared_memory.SharedMemory(create=True, size=(signs.nbytes))

    shm_sizes = [vrec.shape, links.shape, signs.shape]
    shm_names = [shm.name, shm1.name, shm2.name]
    shm_types = [vrec.dtype, links.dtype, signs.dtype]

    nuvrec = np.ndarray(vrec.shape, dtype=vrec.dtype, buffer=shm.buf)
    nuvrec[:] = vrec[:]
    nulinks = np.ndarray(links.shape, dtype=links.dtype, buffer=shm1.buf)

    nulinks[:] = links[:]
    nusign = np.ndarray(signs.shape, dtype=signs.dtype, buffer=shm2.buf)
    nusign[:] = signs[:]


    cdict = dict(zip(allterms, range(numvecs)))
    rdict = dict(zip(range(numvecs),allterms))
    train_file(filename, cdict, tlist, shm_names, shm_sizes, shm_types, numsamples, nsfrq,ssprob,epochs, batchsize, rdict)

    svecs = bv.BinaryVectorStore()
    evecs = bv.BinaryVectorStore()
    pvecs = bv.BinaryVectorStore()

    interms = set(interms)
    inpreds = set(inpreds)

    for key in cdict:
        if key in interms:
            svecs.put_vector(key, bv.BinaryVectorFactory.generate_vector(nusign[cdict[key]]))
        else:
            if key in inpreds:
                pvecs.put_vector(key, bv.BinaryVectorFactory.generate_vector(nusign[cdict[key]]))
            else:
                evecs.put_vector(key, bv.BinaryVectorFactory.generate_vector(nusign[cdict[key]]))

    print('writing vectors')
    svecs.write_vectors('svecs.bin')
    evecs.write_vectors('evecs.bin')
    pvecs.write_vectors('pvecs.bin')

    shm.close()
    shm.unlink()
    shm1.close()
    shm1.unlink()
    shm2.close()
    shm2.unlink()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-filename')
    parser.add_argument('-minfrequency')
    parser.add_argument('-maxfrequency')
    parser.add_argument('-dim')
    parser.add_argument('-numsamples')
    parser.add_argument('-ssthresh')
    parser.add_argument('-bufsize')
    parser.add_argument('-epochs')
    parser.add_argument('-batchsize')
    #parser.add_argument('-update_preds') - to do implement trainable predicate vectors
    args = parser.parse_args()



    main(args)

