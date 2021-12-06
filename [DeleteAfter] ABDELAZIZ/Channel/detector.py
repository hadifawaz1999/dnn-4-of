import numpy as np

class Detector :

    #-------------------------------------------------------------------------------------------------#


    def __init__(self, transmitor):
        self.transmitor = transmitor

    #-------------------------------------------------------------------------------------------------#

    def detector(self, shat, M):
        """ Computes detection based on ML detector """

        cnt, _, _  = self.transmitor.build_constellations(M)
        symboles = []
        indexes = []

        for symb in shat:
            diff = np.absolute(symb - cnt)**2
            index = np.argmin(diff)
            indexes.append(index)
            symboles.append(cnt[index])

        stilde = np.array(symboles)
        return stilde, np.array(indexes)

    #-------------------------------------------------------------------------------------------------#

    def symbols_to_bit(self, indexes, M):

        """ Find the bit associated to the symbol """

        _, _, binaryArr = self.transmitor.build_constellations(M)

        bits = np.ndarray(0, dtype=np.int32)
        for index in indexes:
            bits = np.hstack((bits, binaryArr[index]))
        return bits

    #-------------------------------------------------------------------------------------------------#

    def ber(self, b, bhat):

        """ Computes bit error rate """
        return np.sum(np.abs(b.T - bhat)) / b.size

    #-------------------------------------------------------------------------------------------------#

    def ser(self, s, shat):

        """ Computes symbol error rate """
        return np.sum(np.absolute(s - shat)) / s.size

    #-------------------------------------------------------------------------------------------------#
    