import numpy as np

class Channel :

    #-------------------------------------------------------------------------------------------------#

    def __init__(self):
        super()

    #-------------------------------------------------------------------------------------------------#

    def channel(self, t, q0t, z, sigma2, B) :
        # total noise power in B Hz and distance [0, z]
        a = sigma2*B*z 
        # get the f vector from t vector
        f = np.fft.fftfreq(t.size, 2*t[-1]/t.size)
        omega = 2*np.pi*f
        omega2 = omega**2
        # input in frequency
        q0f = np.fft.fft(q0t)
        #output in frequency
        hwz = np.exp(1j*z*omega2)
        # element-wise multiplication
        qzf = np.multiply(q0f,hwz) 
        #add Guassian noise in frequency, with correct variance
        N = len(f)
        qzf = qzf + np.random.normal(sigma2, np.sqrt(a), qzf.shape)
        # back to time
        qzt = np.fft.ifft(qzf) 

        return qzt, qzf
        
    #-------------------------------------------------------------------------------------------------#

    def channel_(self,t, q0t, z, sigma2, B):
        a = sigma2*B*z  # total noise power in B Hz and distance [0, z]
        f = np.fft.fftfreq(t.size, 2*t[-1]/t.size)  # get the f vector from t vector

        q0f = np.fft.fft(q0t)  # input in frequency
        qzf = q0f * self.H(f, z)  # output in frequency

        # add Guassian noise in frequency, with correct variance
        qzf = qzf + np.random.normal(sigma2, np.sqrt(a), qzf.shape)
        qzt = np.fft.ifft(qzf) # back to time

        return qzt, qzf

    def H(self,f, z):
        w = 2 * np.pi * f
        return np.exp(1j*z*w**2)