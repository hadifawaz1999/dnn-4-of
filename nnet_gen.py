# from channel import Channel
# from params import Params
import numpy as np
from math import *
from scipy.linalg import dft
import scipy

class Generative_Model:
    
    def __init__(self,params):
        
        self.params = params
        
    def noise(self,n,sigma2):
        
        self.z = self.params.Length
        
        self.dz = self.z / self.params.nz
        self.Pn = sigma2 * self.params.Bandwith * self.dz
        noise_real = np.random.normal(loc=0,scale=sqrt(self.Pn/2),size=n)
        noise_imag = np.random.normal(loc=0,scale=sqrt(self.Pn/2),size=n)
        
        noise = np.asarray([complex(noise_real[i],noise_imag[i]) for i in range(n)])
        
        return noise
    
    def activation(self,x,dz):
        
        # print(x.T.dot(x))
        # print(x)
        
        # print(np.exp(1j*2*dz* (x.T).dot(x)).shape)
        
        # return x * np.exp(1j*2*dz* (x.T).dot(x))
        
        return x * np.exp(-1j*2*dz* np.abs(x)**2)
        
        # return x
    
    def nnet_generator(self,x):
        
        n = len(x)
        f = self.params.fft_frequencies
        w = np.fft.fftshift(2*pi*self.params.f)
        
        z = self.params.Length
        dz = 1 / self.params.nz
        
        h = np.exp(1j*w**2 * dz)
        
        # D = dft(n) / np.sqrt(n)
        # 
        # W = ((D.T).conjugate().dot(np.diag(h))).dot(D)
        # 
        
        
        
        for i in range(self.params.nz):
            
            # x = np.dot(W,x)
            
            # x = np.fft.ifft(np.diag(h).dot(np.fft.fft(x)))
            
            x = np.fft.ifft(h * np.fft.fft(x))
            
            x = self.activation(x,dz)
            
            # x = x + self.noise(n=n,sigma2=0)
            
        y = x
        
        return y