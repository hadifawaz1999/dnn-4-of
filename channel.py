import numpy as np
import matplotlib.pyplot as plt
from math import *
import scipy
from data import DataGenerator


class Channel:

    def __init__(self, Length=1e3, Bandwith=6, power_loss_db=0.2*1e-3, dispersion=17, Gamma=1.27,
                 nsp=1, h=6.626*1e-34, lambda0=1.55, T=200, N=2**5, number_symbols=3, p=0.5,M=16):

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
        self.M = M

        self.T = T
        self.N = N
        self.number_symbols = number_symbols  # n in project guide
        self.p = p  # proba of having 0 in bit

        self.data_generator = DataGenerator(

            Bandwith=self.Bandwith, Length=self.Length, power_loss_db=self.power_loss_db,
            dispersion=self.dispersion, Gamma=self.Gamma, nsp=self.nsp, lambda0=self.lambda0,
            T=self.T, N=self.N, number_symbols=self.number_symbols, p=self.p, M=self.M

        )
        
        self.constellation = self.data_generator.Constellation
        self.Constellation = np.asarray([complex(self.constellation[i,0],
                                                  self.constellation[i,1]) for i in range(self.M)])
        
        self.number_bits = self.data_generator.number_bits

        self.data_generator.source()
        self.data_generator.bit_to_symb()
        self.data_generator.mod()

        self.q0t = self.data_generator.q0t

        self.sigma2 = self.data_generator.sigma2

        self.t = self.data_generator.t

        self.fft_frequencies = np.fft.fftfreq(
            n=self.N, d=self.data_generator.dt)

    def setter_function(self, q0t):

        self.q0t = q0t
        
    def setter_noise(self,sigma2):
        
        self.sigma2 = sigma2
        
    def setter_function_nnet(self,qlt):
        
        self.qzte = qlt

    def channel_transfer_function(self, z=None):
        
        if z == None:
            
            z = self.Length

        self.hzf = np.exp(1j*(2*pi*self.fft_frequencies)**2 * z)

        return self.hzf

    def channel(self, z=None):
        
        if z == None:
            
            z = self.Length

        self.z = z
        self.a = self.sigma2 * self.Bandwith * self.z
    

        self.f = self.data_generator.f

        self.q0f = np.fft.fft(self.q0t)

        # Channel Transfer function
        self.qzf = np.multiply(self.q0f, self.channel_transfer_function(z))

        # noise is an (N,) numpy array of complex valued Gaussian(0,sigma2) white noise
        # real part independent of imaginary part

        # noise = np.asarray([complex(np.random.normal(loc=0, scale=sqrt(self.a / 2), size=1),
        #                             np.random.normal(loc=0, scale=sqrt(self.a / 2), size=1))
        #                     for _ in range(self.N)])

        self.qzf += np.random.normal(loc=0,scale=sqrt(self.a),size=self.N)

        self.qzt = np.fft.ifft(self.qzf)

    def equalize(self, z=None):
        
        if z == None:
            
            z = self.Length

        self.z = z

        self.f = self.data_generator.f

        self.qzfe = np.multiply(np.reciprocal(self.hzf), self.qzf)

        self.qzte = np.fft.ifft(self.qzfe)

    def compare(self):

        self.difference = np.mean(np.abs(self.qzte-self.q0t))

        return self.difference < 1e-10

    def dmod(self):

        self.dt = self.data_generator.dt

        self.l1 = -floor(self.number_symbols / 2)
        self.l2 = ceil(self.number_symbols / 2) - 1

        self.s_hat = []

        for i in range(self.l1, self.l2+1, 1):

            temp = sqrt(self.Bandwith * self.T0) * np.sum(np.multiply(self.qzte,
                                         np.sinc(self.Bandwith * self.T0 *self.t - i))) * self.dt
            
            self.s_hat.append(temp)
            
        self.s_hat = np.asarray(self.s_hat)

        
    def detector(self):
        
        
        self.s_tilde = []
        
        for i in range(self.number_symbols):
            
            distance_to_constellation = np.abs(self.Constellation - self.s_hat[i])**2
            
            index_min_distance = np.argmin(distance_to_constellation)
            
            self.s_tilde.append(self.Constellation[index_min_distance])
            
        self.s_tilde = np.asarray(self.s_tilde)
        
    
    def symb_to_bit(self):
        
        self.gray_to_symb = self.data_generator.gray_to_symb
        
        self.symb_to_gray = {complex(v[0],v[1]) : k for k , v in self.gray_to_symb.items()}
        
        self.b_hat = []
        
        for i in range(self.number_symbols):
            
            self.b_hat.append(self.symb_to_gray[self.s_tilde[i]])
            
        self.b_hat = np.asarray(self.b_hat)
        
    def test_bs(self):
        
        self.data_generator.testbs()
        self.qzte = self.data_generator.q0t
        self.number_symbols=2
        self.dmod()
        
        
        print(self.s_hat)
        
    def evaluate_results(self):
        
        self.s = self.data_generator.s
        
        self.s = np.asarray([complex(self.s[i,0],self.s[i,1]) for i in range(len(self.s))])
        
        self.b = self.data_generator.bernoulli
        
        self.b_string = []
        
        b0=b1=b2=b3=""
        
        for i in range(0, self.number_bits, int(log2(self.M))):
            
            b0 = str(self.b[i])
            
            if int(log2(self.M)) > 1:
                b1 = str(self.b[i+1])
                if int(log2(self.M)) > 2:
                    b2 = str(self.b[i+2])
                    if int(log2(self.M)) > 3:
                        b3 = str(self.b[i+3])
            
            b_i = b0+b1+b2+b3

            self.b_string.append(b_i)
        
        self.b_string = np.asarray(self.b_string)
        
        return self.find_score_symbs(self.s,self.s_tilde) , self.find_score_bits(self.b_string,self.b_hat)
    
    def find_score_bits(self,a,b):
        
        score = 0
        
        for i in range(len(a)):
            
            for j in range(int(log2(self.M))):
                
                if str(a[i])[j] == str(b[i])[j]:
                    
                    score += 1
        
        return score / self.number_bits
    
    def find_score_symbs(self,a,b):
        
        
        a.shape = (-1,)
        b.shape = (-1,)
        
        score = 0
        
        for i in range(len(a)):
            
            if a[i] == b[i]:
                
                score += 1
        
        return score / self.number_symbols