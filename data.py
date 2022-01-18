from math import *
import numpy as np
import matplotlib.pyplot as plt
import scipy


class DataGenerator:

    def __init__(self, Length=1e6, Bandwith=6, power_loss_db=0.2*1e-3, dispersion=17, Gamma=1.27*1e-3,
                 nsp=1, h=6.626*1e-34, lambda0=1.55, T=200, N=2**5, number_symbols=3, p=0.5,M=16,):

        self.Length = Length
        self.Bandwith = Bandwith # GHz to Hz
        self.power_loss_db = power_loss_db
        self.dispersion = dispersion
        self.Gamma = Gamma
        self.nsp = nsp
        self.h = h
        self.lambda0 = lambda0
        self.c = 3e8
        self.f0 = self.c / self.lambda0
        self.alpha = 1e-4 * log2(10) * self.power_loss_db
        self.beta2 = - (self.lambda0**2 / (2 * pi * self.c)) * self.dispersion
        self.L0 = self.Length
        self.T0 = sqrt(abs(self.beta2)*self.L0 / 2)
        self.P0 = 2 / (self.Gamma * self.L0)
        
        self.Bandwith_n = self.Bandwith * self.T0
        
        
        self.sigma02 = self.nsp * self.h * self.alpha * self.f0
        self.sigma2 = (self.sigma02 * self.L0) / (self.P0 * self.T0)
        self.M = M
        
        self.T = T
        self.N = N
        self.number_symbols = number_symbols # n in project guide
        self.p = p # proba of having 0 in bit
    
        
        self.dt = self.T / self.N

        self.t = np.arange(start=-self.T/2,stop=self.T/2,step=self.dt)
        
        self.F = 1 / self.dt

        self.df = 1 / self.T

        self.f = np.arange(start=-self.F/2,stop=self.F/2,step=self.df)
        
        self.fft_frequencies = np.fft.fftfreq(n=self.N,d=self.dt)
        
        self.number_bits = self.number_symbols * int(log2(self.M)) # nb in guide

        # Constellation

        if M == 16:

            self.Constellation = np.zeros(shape=(16, 2))

            points_1 = [-3, -1, 1, 3]
            points_2 = [3, 1, -1, -3]

            const1 = np.asarray([[i, 3] for i in points_1])
            const2 = np.asarray([[i, 1] for i in points_2])
            const3 = np.asarray([[i, -1] for i in points_1])
            const4 = np.asarray([[i, -3] for i in points_2])

            self.Constellation = np.concatenate((np.concatenate((const1, const2), axis=0),
                                                np.concatenate((const3, const4), axis=0)), axis=0)
        else:
            
            self.generate_constellation(M=self.M)
            
        self.average_constellation_power = np.mean(self.Constellation[:,0]**2 + self.Constellation[:,1]**2)
        
        self.Constellation = np.divide(self.Constellation,self.average_constellation_power)

    def generate_constellation(self,M):
        
        a = 1
        
        if M == 2:
            
            self.Constellation = np.array([[-a,0],[a,0]])
        
        elif M == 4:
            
            self.Constellation = np.array([[-a,-a],[a,a],[-a,a],[a,-a]])
            
        elif M == 8:
            
            self.Constellation = np.array([[-a,-a],[a,a],[-a,a],[a,-a],
                                           [(a+sqrt(3))*a,0],[-(a+sqrt(3))*a,0],
                                           [0,(a+sqrt(3))*a],[0,-(a+sqrt(3))*a]])
    
    def source(self):


        self.bernoulli = np.random.binomial(n=1, p=self.p, size=self.number_bits)

        # return self.bernoulli

    def bit_to_symb(self):

        self.gray_code = []

        for i in range(0, 1 << int(log2(self.M))):

            gray = i ^ (i >> 1)
            self.gray_code.append("{0:0{1}b}".format(gray, int(log2(self.M))))

        self.gray_code = np.asarray(self.gray_code)
        
        self.gray_to_symb = dict(zip(self.gray_code,self.Constellation))
        

        self.s = []


        b0=b1=b2=b3=""

        for i in range(0, self.number_bits, int(log2(self.M))):
            
            b0 = str(self.bernoulli[i])
            
            if int(log2(self.M)) > 1:
                
                b1 = str(self.bernoulli[i+1])
                
                if int(log2(self.M)) > 2:
                    
                    b2 = str(self.bernoulli[i+2])
                    
                    if int(log2(self.M)) > 3:
                        
                        b3 = str(self.bernoulli[i+3])
            
            b_i = b0+b1+b2+b3


            self.s.append(self.gray_to_symb[b_i])
        
        self.s = np.asarray(self.s)

    def mod(self):
        
        Ns = len(self.s)
        
        self.l1 = -floor(Ns / 2)
        
        self.l2 = ceil(Ns/2) - 1
        
        self.q0t_list = []
        
        for i in range(self.l1,self.l2+1,1):
            
            self.q0t_list.append(sqrt(self.Bandwith_n) * complex(self.s[i - self.l1,0],self.s[i - self.l1,1]) *\
                                 np.sinc(self.Bandwith_n * self.t - i))
        
        self.q0t_list = np.asarray(self.q0t_list)
        
        self.q0t = np.sum(self.q0t_list,axis=0)
        
        self.q0t_FFT = np.fft.fft(self.q0t)   
        
        
    def plot_q0t(self):
        
        self.q0t_norm = [sqrt(self.q0t[i].real**2 + self.q0t[i].imag**2) for i in range(self.N)]

        plt.plot(self.t,self.q0t_norm,color='black')
        plt.ylabel(r'|q(0,t)|')
        plt.xlabel(r'time')
        plt.title('Norm of '+r'q(0,t)'+' vs time with Bandwith = '+str(self.Bandwith))
        plt.savefig('./plots/norm_q0t.png')
        plt.clf() 

        self.fft_frequencies = np.fft.fftfreq(n=self.N,d=self.dt)
        
        plt.plot(self.fft_frequencies,np.abs(self.q0t_FFT),color='red')
        plt.xlabel(r'frequency')
        plt.ylabel(r'FFT(q(0,t))')
        plt.title(r'q(0,t)'+'in frequency domain with Bandwith = '+str(self.Bandwith))
        plt.savefig('./plots/fft_q0t.png')


    def draw_Constellation(self):

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.spines['left'].set_position('center')
        ax.spines['bottom'].set_position('center')
        ax.spines['right'].set_color('none')
        ax.spines['top'].set_color('none')

        plt.title(str(self.M)+'-QAM constellation', pad=20)

        for i in range(self.M):
            plt.scatter(self.Constellation[i, 0],
                        self.Constellation[i, 1], color='red')
            if self.Constellation[i, 1] > 0:
                plt.annotate(text=str(self.Constellation[i, 0])+' + j'+str(self.Constellation[i, 1]),
                             xy=(self.Constellation[i, 0],
                                 self.Constellation[i, 1]),
                             xytext=(self.Constellation[i, 0], self.Constellation[i, 1]+0.2))
            else:
                plt.annotate(text=str(self.Constellation[i, 0])+' - j'+str(abs(self.Constellation[i, 1])),
                             xy=(self.Constellation[i, 0], self.Constellation[i, 1]), ha='center', va='center',
                             xytext=(self.Constellation[i, 0], self.Constellation[i, 1]+0.2))

        plt.savefig("plots/Constellation.png")
