import numpy as np
from numba import jitclass
from numba import uint8, uint32, float32


spec = [
    ('dimension', uint32),
    ('bitset', uint8[:]),               # a simple scalar field
    ('votingRecord', float32[:]),          # an array field
]


@jitclass(spec)
class BinaryVector(object):

    def __init__(self, dimension):
        self.dimension = dimension
        self.bitset, self.votingRecord = self.generate_random_vector()

    #Someday it will probably make sense to make a packedbits implementation
    def packbits(self):
        packedarray = np.zeros(uint32(self.dimension/8), dtype=np.uint8)
        for offset in range(0, self.dimension, 8):
            packed = 0
            bits = self.bitset[offset:offset+8][::-1]
            for i in range(8):
                if bits[i]:
                    packed += 2**i
            packedarray[uint32(offset/8)] = packed
        return packedarray

    # copy bitset and voting record
    def copy(self):
        new = Vector(self.dimension)
        new.bitset = self.bitset.copy() # replaces 0 bitset
        new.votingRecord = self.votingRecord.copy() # replaces 0 votingRecord
        return new

    # randomly make half of the bit set to 1, voting record is same as bitset (vote for yourself)
    def generate_random_vector(self):
        halfdimension = uint32(self.dimension/2)
        randvec = np.concatenate((np.ones(halfdimension, dtype=uint8), np.zeros(halfdimension, dtype=uint8)))
        np.random.shuffle(randvec)
        self.bitset = randvec
        self.votingRecord = randvec.astype(float32)
        #return randvec, randvec.astype(float32)
        # This is only 1-2us faster than the above code JIT'd
        # but either option JIT'd is ~30x faster than straight python
        return self.bitset, self.votingRecord

    # make bitset all zeros, voting record is all zeros
    def generate_zero_vector(self):
        self.bitset = np.zeros(self.dimension, dtype=uint8)
        self.votingRecord = np.zeros(self.dimension,dtype=float32)

    def get_dimension(self):
        return self.dimension

    def get_vector_type(self):
        return self.bitset.dtype

    def is_zero_vector(self):
        return not self.bitset.any()

    # Non negative overlap, normalized for dimensions
    def measure_overlap(self,other):
        return 1 - (self.bitset ^ other.bitset).sum()/(self.dimension)

    # superpose other to self with weight (update voting record of self)
    def superpose(self, other, weight):
        for i in range(self.dimension):
            if other.bitset[i]:
                self.votingRecord[i] += weight
            else:
                self.votingRecord[i] -= weight

    # update bitset as xor of self bitset and other bitset
    def bind(self, other):
        self.bitset =  self.bitset ^ other.bitset

    # update bitset as xor of self bitset and other bitset
    def release(self, other):
        self.bitset = self.bitset ^ other.bitset

    # return the majority rule voting from voting record (return a 0/1 boolean vector)
    def tallyVotes(self):
        if np.sum(np.abs(self.votingRecord)) == 0: return #this shortcut only occurs after a normalization
        s = np.sign(self.votingRecord)
        s[s==0] = np.random.choice(np.array([-1,1]),s[s==0.].shape[0]) #make as many random checks as their are 0s
        s[s==-1] = 0 #set all negatives to 0
        self.bitset = s.astype(np.uint8) # copy updated record to bitset as uint8

    # tallyVotes (and update the bitset) and zeros out the voting record
    def normalize(self):
        self.tallyVotes() 
        self.votingRecord = np.zeros(self.dimension,dtype=float32)


