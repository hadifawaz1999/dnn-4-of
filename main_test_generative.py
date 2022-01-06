from data import DataGenerator
from nnet_gen import Generative_Model
from params import Params
import numpy as np

params = Params(N=2**8,nz=1)
dg = DataGenerator(N=2**8)

dg.source()
dg.bit_to_symb()
dg.mod()

NNET_GEN = Generative_Model(params=params)

x = dg.q0t

x /= (np.sum(np.abs(x)**2))

print(np.sum(np.abs(x)**2))

y = NNET_GEN.nnet_generator(x=x)

# print(y)