import numpy as np
from math import *


class Params:
    
    def __init__(self, Length=1000e3, Bandwith=10e9, power_loss_db=0.2*1e-3, dispersion=17*1e-6, Gamma=1.27*1e-6,
                 nsp=1, h=6.626*1e-34, lambda0=1.55*1e-6, T=40, N=2**10, number_symbols=3, p=0.5,M=16,
                 nz=500):
        
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
        #self.sigma02 = self.nsp * self.h * self.alpha * self.f0

        self.sigma02 = 3.16e-20

        self.sigma2 = (self.sigma02 * self.L0) / (self.P0 * self.T0)
        self.M = M

        self.T = T
        self.N = N
        self.number_symbols = number_symbols  # n in project guide
        self.p = p  # proba of having 0 in bit
        
        self.dt = self.T / self.N

        self.t = np.arange(start=-self.T/2,stop=self.T/2,step=self.dt)
        
        self.F = 1 / self.dt

        self.df = 1 / self.T

        self.f = np.arange(start=-self.F/2,stop=self.F/2,step=self.df)
        
        self.fft_frequencies = np.fft.fftfreq(n=self.N,d=self.dt)
        
        self.number_bits = self.number_symbols * int(log2(self.M)) # nb in guide
        
        self.nz = nz
        