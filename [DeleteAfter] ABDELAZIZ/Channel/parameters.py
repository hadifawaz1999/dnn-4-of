import numpy as np
from scipy.linalg import dft
import scipy.constants as scipy_cst

class Parameters :
    
    
    def __init__(self, bandwidth = 1, nsymbols=16, M = 16, Nt= 2**10, nLayers=100, timeMesh=200):
        
        # distance variables are converted in km
        #--------------------------------- variables ------------------------------------------------------#

        # distance
        self.L = 1#1000e3
        # bandwidth - 1 GHz
        self.B = bandwidth

        #---------------------------------- physical constants --------------------------------------------#

        # power loss in dB
        self.a_dB = 0.2 
        # loss coefficient
        self.alpha = (1e-4)*np.log2(10)*self.a_dB
        # speed of light m/s
        self.c = 3e8
        # dispersion ps/(nm-m)
        self.D = 17*1e-6
        #non-linearity coefficient - W^-1 m^-1
        self.gama = 1.27*1e-3
        # a constant factor
        self.nsp = 1
        # Planck constant - J.s
        self.h = scipy_cst.Planck
        # center wavelength - m
        self.lambda0 = 1.55e-6 #1.55e-9
        # center frequency
        self.f0 = self.c/self.lambda0    
        # dispersion coefficient
        self.beta2 = -(self.D*(self.lambda0**2))/(2*np.pi*self.c)

        #---------------------------------- scale factors ------------------------------------------------#

        self.L0 = self.L
        self.T0 = np.sqrt(np.abs(self.beta2*self.L)/2)
        self.P0 = 2/(self.gama*self.L)

        #----------------------------------- noise PSD ---------------------------------------------------#

        # physical noise 
        self.sigma02 = self.nsp*self.h*self.alpha*self.f0
        # normalized
        self.sigma2 = self.sigma02*self.L/(self.P0*self.T0) 

        #-------------------------------------------------------------------------------------------------#

        self.power = 6e-2
        self.l = 1#5
        self.nz = nLayers # 500
        self.z = 1#self.l/self.L #1

        #------------------------------------ time mesh --------------------------------------------------#

        # you have to choose this
        self.T = timeMesh #( (1/bandwidth)*self.nb ) + (10*2/ self.nb) #timeMesh # 200#20 # (1/B)*Ns + 10*2/B
        # you have to choose this
        self.N = Nt #2*10**2 #2**10
        self.dt = self.T/self.N
        self.t = np.linspace(-self.T/2, self.T/2, num=self.N, endpoint=True)

        #------------------------------------ frequency mesh ---------------------------------------------#

        self.F = 1/self.dt
        self.df = 1/self.T
        self.f = np.linspace(-self.F/2, self.F/2, num=self.N, endpoint=True) #np.fft.fftfreq(self.N, self.dt)
        self.omega = 2*np.pi*self.f
        self.w = np.fft.fftshift(self.omega)

        #------------------------------------ bits to signal ---------------------------------------------#

        # size of the constellation
        self.M = M
        # number of symbols (or sinc functions); test with s=1
        self.n = nsymbols
        # number of bits
        self.nb = int(self.n * np.log2(self.M))
        # probability of zero
        self.p = 1/2
        
        #------------------------------------ Generative nnet ---------------------------------------------#
        
        # Number of the Layers of the Generative nnet 
        self.nbrLayers = nLayers
        # epsilon
        self.eps = self.z / self.nz
        
        # Amplitude for AWGN
        self.A = 1
        self.s = 2
    
    def setN(self,n):
        self.N = n
        self.__init__(N=n)
    
    def setNbrLayers(self,N):
        self.nbrLayers = N
        self.__init__(nbrLayers=N)