from math import *
import numpy as np
import matplotlib.pyplot as plt


class DataGenerator:

    def __init__(self, Length=1e6, Bandwith=5, power_loss_db=0.2*1e-3, dispersion=17*1e-3, Gamma=1.27*1e-3,
                 nsp=1, h=6.626*1e-34, lambda0=1.55*1e-6):

        self.Length = Length
        self.Bandwith = Bandwith
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
        self.sigma02 = self.nsp * self.h * self.alpha * self.f0
        self.sigma2 = (self.sigma02 * self.L0) / (self.P0 * self.T0)
        self.M = 16

        # Constellation

        self.Constellation = np.zeros(shape=(16, 2))

        points_1 = [-3, -1, 1, 3]
        points_2 = [3, 1, -1, -3]

        const1 = np.asarray([[i, 3] for i in points_1])
        const2 = np.asarray([[i, 1] for i in points_2])
        const3 = np.asarray([[i, -1] for i in points_1])
        const4 = np.asarray([[i, -3] for i in points_2])

        self.Constellation = np.concatenate((np.concatenate((const1, const2), axis=0),
                                             np.concatenate((const3, const4), axis=0)), axis=0)

    def source(self, N, p):

        self.N = N
        self.p = p

        self.bernoulli = np.random.binomial(n=1, p=self.p, size=self.N)

        # return self.bernoulli

    def bit_to_symb(self):

        self.n = self.N / log2(self.M)

        self.gray_code = []

        for i in range(0, 1 << int(log2(self.M))):

            gray = i ^ (i >> 1)
            self.gray_code.append("{0:0{1}b}".format(gray, int(log2(self.M))))

        self.gray_code = np.asarray(self.gray_code)
        
        self.gray_to_symb = dict(zip(self.gray_code,self.Constellation))

        self.s = []


        for i in range(0, self.N, 4):
            
            b0 = str(self.bernoulli[i])
            b1 = str(self.bernoulli[i+1])
            b2 = str(self.bernoulli[i+2])
            b3 = str(self.bernoulli[i+3])
            
            b_i = b0+b1+b2+b3


            self.s.append(self.gray_to_symb[b_i])
        


    def draw_Constellation(self):

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.spines['left'].set_position('center')
        ax.spines['bottom'].set_position('center')
        ax.spines['right'].set_color('none')
        ax.spines['top'].set_color('none')

        plt.title('16-QAM constellation', pad=20)

        for i in range(16):
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

        plt.savefig("Constellation.png")
