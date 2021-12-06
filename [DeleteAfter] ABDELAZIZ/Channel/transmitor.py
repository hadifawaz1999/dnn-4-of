import numpy as np
from matplotlib.ticker import MultipleLocator
import matplotlib.pyplot as plt

class Transmitor :

    def __init__(self):
        super()


    #-------------------------------------------------------------------------------------------------#
    
    def source(self, N, p):
        """ Generates an N-bit sequence drawn from a Bernoulli stochastic process.
            @param N : Length of the bit sequence
            @param p : probability of 0
        """
        # number of experiences
        n=1
        return np.random.binomial(n, p, size=N)

    #-------------------------------------------------------------------------------------------------#

    def build_constellations(self,M):
        k = int(np.log2(M))
        N = np.arange(0, M, dtype=np.int32)
        coords = []
        binaryStr = []
        binaryArr = np.zeros((M, k), dtype=np.uint8)
        for n in N:
            bin, code = self.gray_coding(n, M)
            coords.append(code)
            binaryArr[n] = bin
            binaryStr.append(np.binary_repr(n, width=k))

        coords = np.array(coords)

        return coords, binaryStr, binaryArr

    #-------------------------------------------------------------------------------------------------#

    def bit_to_symb(self, b, M=16):
        """ Creates a mapping between bits sequences and symbols.
            @param b : N-bit sequence
            @param M : Number of symbols in the constellation
        """

        k = int(np.log2(M))

        if b.size % k != 0:
            b = np.vstack((b, np.zeros((6-b.size % k, 1), dtype=np.uint8)))

        bits = b.reshape((-1, k))

        symboles = []
        for bi in bits:
            biDec = np.packbits(np.hstack((np.zeros(8-k, dtype=np.uint8), bi)))[0]
            bin, code = self.gray_coding(biDec, M)
            symboles.append(code)

        return np.array(symboles)

    #-------------------------------------------------------------------------------------------------#

    def gray_coding(self, n, M):
        k = int(np.log2(M))

        reAxis = np.hstack((np.arange(-(np.sqrt(M)//2), 0, step=1),
                            np.arange(1, np.sqrt(M)/2+1, step=1)))
        imAxis = np.copy(reAxis)

        bin = np.unpackbits(np.array([n], dtype=np.uint8))[8-k:]

        rePart = bin[:k//2]
        imPart = bin[k//2:]

        aReBin = np.hstack((np.zeros(8-k//2, dtype=np.uint8), rePart))
        aImBin = np.hstack((np.zeros(8-k//2, dtype=np.uint8), imPart))

        aReDec = np.packbits(aReBin)
        aImDec = np.packbits(aImBin)

        reIndex = np.bitwise_xor(aReDec, aReDec//2)[0]
        imIndex = np.bitwise_xor(aImDec, aImDec//2)[0]

        return  bin, complex(reAxis[reIndex], imAxis[imIndex])
    
    #-------------------------------------------------------------------------------------------------#

    def plot_constellation(self, M, constellation, binaryStr):

        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111)
        fig.suptitle(f'{M}-QAM Constellation')
        ax.set_xlabel('Real part')
        ax.set_ylabel('Imaginary part')

        ax.scatter(constellation.real, constellation.imag)

        for i, str_bin in enumerate(binaryStr):
            ax.annotate(str_bin, (constellation.real[i], constellation.imag[i]),
                        xytext=(constellation.real[i]-0.2, constellation.imag[i]-0.2))

        ax.axis([-(np.sqrt(M)//2)-1, np.sqrt(M)/2+1, -(np.sqrt(M)//2)-1, np.sqrt(M)/2+1])
        ax.axhline(y=0, color='black')
        ax.axvline(x=0, color='black')

        ax.xaxis.set_major_locator(MultipleLocator(1))
        ax.yaxis.set_major_locator(MultipleLocator(1))
        
        plt.show()

    #-------------------------------------------------------------------------------------------------#
    
    def build_constellations_2(self, M):
    
        """ Builds a M-QAM constellation.
            @param M : Number of symbols.
        """
        
        # Sequential address from 0 to M-1 (1xM dimension)
        n = np.arange(0,M)
        #convert linear addresses to Gray code
        a = np.asarray([x^(x>>1) for x in n])
        #Dimension of K-Map - N x N matrix
        D = np.sqrt(M).astype(int) 
        # NxN gray coded matrix
        a = np.reshape(a,(D,D))
        # identify alternate rows
        oddRows=np.arange(start = 1, stop = D ,step=2) 
        
        # reshape to 1xM - Gray code walk on KMap
        nGray=np.reshape(a,(M)) 
        
        #element-wise quotient and remainder
        (x,y)=np.divmod(nGray,D) 
        # PAM Amplitudes 2d+1-D - real axis
        Ax=2*x+1-D 
        # PAM Amplitudes 2d+1-D - imag axis
        Ay=2*y+1-D 
        constellation = Ax+1j*Ay
        
        self.constellation = constellation

        return constellation


    #-------------------------------------------------------------------------------------------------#
