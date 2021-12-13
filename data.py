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
        self.sigmoid02 = self.nsp * self.h * self.alpha * self.f0
        self.sigma2 = (self.sigma2 * self.L0) / (self.P0 * self.T0)
        self.M = 6

        # Constellation

        self.Constellation = np.zeros(shape=(16, 2))

        points = [-3, -1, 1, 3]

        const1 = np.asarray([[i, 3] for i in points])
        const2 = np.asarray([[i, 1] for i in points])
        const3 = np.asarray([[i, -1] for i in points])
        const4 = np.asarray([[i, -3] for i in points])

        self.Constellation = np.concatenate( ( np.concatenate( (const1, const2), axis=0),
                                             np.concatenate((const3, const4), axis=0) ),axis=0 )

        plt.scatter(self.Constellation[:,0],self.Constellation[:,1])
        plt.savefig('Constellation.png')