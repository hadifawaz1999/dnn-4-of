import numpy as np

class Modulator :

    #-------------------------------------------------------------------------------------------------#

    def __init__(self):
      super()

    #-------------------------------------------------------------------------------------------------#
    
    def mod(self, t, s, B):

        """ Modulate symbols data.
            @param t : time.
            @param s : sequence of symbols.
            @param B : Bandwidth.
        """

        Ns = len(s)  # Number of Symbols

        l1 = int(-(Ns/2))
        l2 = int((Ns/2) - 1)

        lRange = np.linspace(l1, l2, num=Ns, endpoint=True)

        q0t = np.zeros((1, t.size), dtype=np.complex64) #q0t = 0

        for sl, l in zip(s, lRange):
            q0t += sl * np.sinc(B*t - l)

        return np.sqrt(B)*q0t

    #-------------------------------------------------------------------------------------------------#
