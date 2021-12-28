import numpy as np
import matplotlib.pyplot as plt
from math import *

import scipy
from data import DataGenerator


class Channel:

    def __init__(self, Length=1e6, Bandwith=6, power_loss_db=0.2*1e-3, dispersion=17*1e-3, Gamma=1.27*1e-3,
                 nsp=1, h=6.626*1e-34, lambda0=1.55*1e-6, T=200, N=2**5, number_symbols=3, p=0.5):

        self.Length = Length
        self.Bandwith = Bandwith
        self.power_loss_db = power_loss_db
        self.dispersion = dispersion
        self.Gamma = Gamma
        self.nsp = nsp
        self.lambda0 = lambda0
        self.h = h
        self.c = 3e8
        self.f0 = self.c / self.lambda0
        self.alpha = 1e-4 * log2(10) * self.power_loss_db
        self.beta2 = - (self.lambda0**2 / (2 * pi * self.c)) * self.dispersion
        self.L0 = self.Length
        self.T0 = sqrt(abs(self.beta2)*self.L0 / 2)
        self.P0 = 2 / (self.Gamma * self.L0)
        self.sigma02 = self.nsp * self.h * self.alpha * self.f0
        self.sigma2 = (self.sigma02 * self.L0) / (self.P0 * self.T0)
        self.M = 16

        self.T = T
        self.N = N
        self.number_symbols = number_symbols  # n in project guide
        self.p = p  # proba of having 0 in bit

        self.data_generator = DataGenerator(

            Bandwith=self.Bandwith, Length=self.Length, power_loss_db=self.power_loss_db,
            dispersion=self.dispersion, Gamma=self.Gamma, nsp=self.nsp, lambda0=self.lambda0,
            T=self.T, N=self.N, number_symbols=self.number_symbols, p=self.p

        )

        self.data_generator.source()
        self.data_generator.bit_to_symb()
        self.data_generator.mod()

        self.q0t = self.data_generator.q0t

        self.sigma2 = self.data_generator.sigma2

        self.t = self.data_generator.t
    
    def setter_function(self,q0t):
        
        self.q0t = q0t

    def channel_transfer_function(self, z):

        hzf = []

        for i in range(self.N):

            hzf.append(np.exp(complex(0, np.square(2*pi*self.f[i])*z)))

        hzf = np.asarray(hzf)

        return hzf
    
    def channel_inverse_transfer_function(self,z):
        
        hzf = self.channel_transfer_function(z)
        
        hzf_inverse = []
        
        for i in range(self.N):
            
            hzf_inverse.append(complex(0,-1) * np.ln(hzf[i]) / (2*pi*self.f[i])**2 )
            
        return hzf_inverse

    def channel(self, z):

        self.z = z
        self.a = self.sigma2 * self.Bandwith * self.z

        self.f = self.data_generator.f

        self.q0f = self.data_generator.q0t_FFT

        # Channel Transfer function
        self.qzf = self.q0f * self.channel_transfer_function(z)

        # noise is an (N,) numpy array of complex valued Gaussian(0,sigma2) white noise
        # real part independent of imaginary part
        
        noise = np.asarray([complex(np.random.normal(loc=0, scale=self.a / 2, size=1), 
                                    np.random.normal(loc=0, scale=self.a / 2, size=1))
                                    for _ in range(self.N)])

        self.qzf += scipy.fft.fft(noise)
        
        self.qzt = scipy.fft.ifft(self.qzf)

    def equilize(self,z):
        
        self.z = z
        
        self.f = self.data_generator.f
        
        self.qzfe = np.multiply(self.channel_inverse_transfer_function(z),self.qzf)
        
        self.qzte = scipy.fft.ifft(self.qzfe)