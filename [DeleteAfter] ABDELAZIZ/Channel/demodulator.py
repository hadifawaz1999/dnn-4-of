import numpy as np

class Demodulator :

    #-------------------------------------------------------------------------------------------------#

    def demod(self, t, dt, qzte, B, Ns):
        sl = np.zeros((Ns, 1), dtype=np.complex128)
        l1 = int(-(Ns/2))
        l2 = int((Ns/2) - 1)
        
        lRange = np.linspace(l1, l2, num=Ns, endpoint=True)

        for j, l in enumerate(lRange):
            for i, ti in enumerate(t):
                sl[j] += qzte[0, i] * np.sinc(B*ti-l) * dt

        shat = np.sqrt(B)*sl

        return shat

    #-------------------------------------------------------------------------------------------------#
    