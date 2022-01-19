import numpy as np
from data import DataGenerator
from nnet_gen import Generative_Model
from params import Params

import time

start_time = time.time()

dg = DataGenerator(T=200,N=2**10,number_symbols=32,Length=30)

dg.setter_noise(sigma2=10**-4)

params = Params(N=2**10,T=200,number_symbols=32,nz=500,Length=30)
nnet_gen = Generative_Model(params=params)

n_samples = 10**4

feature_vectors = []
bit_signals = []
labels = []
symbols = []


for i in range(n_samples):
    
    dg.source()
    dg.bit_to_symb()
    dg.mod()
    
    x = dg.q0t

    symbols.append(dg.s)
    
    bit_signals.append(dg.bernoulli)
    
    x_flatten = np.asarray([[x[i].real , x[i].imag] for i in range(len(x))]).flatten()
    
    labels.append(x_flatten)
    
    y = nnet_gen.nnet_generator(x=x)   
    
    y_2d = np.asarray([[y[i].real , y[i].imag] for i in range(len(y))])
    
    feature_vectors.append(y_2d)

feature_vectors = np.asarray(feature_vectors)
bit_signals = np.asarray(bit_signals)
labels = np.asarray(labels)
symbols = np.asarray(symbols)

print(time.time() - start_time)

print(feature_vectors.shape)
print(labels.shape)
print(bit_signals.shape)
print(symbols.shape)

fv_train =  feature_vectors[0:7000]
fv_test = feature_vectors[7000:9000]
fv_val = feature_vectors[9000:]

bs_train = bit_signals[0:7000]
bs_test = bit_signals[7000:9000]
bs_val = bit_signals[9000:]

lb_train = labels[0:7000]
lb_test = labels[7000:9000]
lb_val = labels[9000:]

sb_train = symbols[0:7000]
sb_test = symbols[7000:9000]
sb_val = symbols[9000:]

np.save(arr=fv_train,file="data/feature_vectors_train_10knoise.npy")
np.save(arr=fv_test,file="data/feature_vectors_test_10knoise.npy")
np.save(arr=fv_val,file="data/feature_vectors_val_10knoise.npy")

np.save(arr=bs_train,file="data/bit_signals_train_10knoise.npy")
np.save(arr=bs_test,file="data/bit_signals_test_10knoise.npy")
np.save(arr=bs_val,file="data/bit_signals_val_10knoise.npy")

np.save(arr=lb_train,file="data/labels_train_10knoise.npy")
np.save(arr=lb_test,file="data/labels_test_10knoise.npy")
np.save(arr=lb_val,file="data/labels_val_10knoise.npy")

np.save(arr=sb_train,file="data/symbols_train_10knoise.npy")
np.save(arr=sb_test,file="data/symbols_test_10knoise.npy")
np.save(arr=sb_val,file="data/symbols_val_10knoise.npy")
