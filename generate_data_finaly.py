import numpy as np
from data import DataGenerator
from nnet_gen import Generative_Model
from params import Params

import time

start_time = time.time()

dg = DataGenerator(T=200,N=2**10,number_symbols=32)

params = Params(N=2**10,T=200,number_symbols=32,nz=500)
nnet_gen = Generative_Model(params=params)

n_samples = 1e4

feature_vectors = []
bit_signals = []
labels = []

for i in range(n_samples):
    
    dg.source()
    dg.bit_to_symb()
    dg.mod()
    
    x = dg.q0t
    
    bit_signals.append(dg.bernoulli)
    
    x_flatten = np.asarray([[x[i].real , x[i].imag] for i in range(len(x))]).flatten()
    
    labels.append(x_flatten)
    
    y = nnet_gen.nnet_generator(x=x)   
    
    y_2d = np.asarray([[y[i].real , y[i].imag] for i in range(len(y))])
    
    feature_vectors.append(y_2d)

feature_vectors = np.asarray(feature_vectors)
bit_signals = np.asarray(bit_signals)
labels = np.asarray(labels)

print(time.time() - start_time)

print(feature_vectors.shape)
print(labels.shape)
print(bit_signals.shape)

np.save(arr=feature_vectors,file='data/feature_vectors.npy')
np.save(arr=bit_signals,file='data/bit_signals.npy')
np.save(arr=labels,file='data/labels.npy')