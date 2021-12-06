import numpy as np

class Equalizer :

    #-------------------------------------------------------------------------------------------------#

    def __init__(self):
        super()

    #-------------------------------------------------------------------------------------------------#

    def equalize(self, t, qzt, z):
        
        # get the f vector from t vector
        f = np.fft.fftfreq(t.size, 2*t[-1]/t.size)
        # get the omega vector from f vector
        omega = 2*np.pi*f

        # compute omega² for h
        omega2 = omega**2

        # Channel transfer function
        hwz = np.exp(1j*z*omega2)
        hwz_1 = np.reciprocal(hwz)
        # input in frequency
        qzf = np.fft.fft(qzt)
        # output in frequency        
        qzfe = qzf*hwz_1
        # back to time
        qzte = np.fft.ifft(qzfe)
        

        return qzte, qzfe

    def H(self,f, z):
        w = 2 * np.pi * f
        return np.exp(1j*z*w**2)

    def Hinv(self,f, z):
        return np.reciprocal(self.H(f, z))
    #-------------------------------------------------------------------------------------------------#
    def equalize_(self,t, qzt, z):
        f = np.fft.fftfreq(t.size, 2*t[-1]/t.size)  # get the f vector from t vector

        qzf = np.fft.fft(qzt)  # input in frequency
        qzfe = qzf * self.Hinv(f, z)  # output in frequency

        qzte = np.fft.ifft(qzfe) # back to time

        return qzte, qzfe