import numpy as np
from FFN import FFN
from sklearn.model_selection import train_test_split
import time
from channel import Channel
from data import DataGenerator
import matplotlib.pyplot as plt

# feature_vectors = np.load(file='/app/data/feature_vectors_10knoise.npy')
# bit_signals = np.load(file='/app/data/bit_signals_10knoise.npy')
# labels = np.load(file='/app/data/labels_10knoise.npy')

# print(feature_vectors.shape)
# print(labels.shape)

xtrain = np.load('/app/data/feature_vectors_train_10knoise.npy')
ytrain = np.load('/app/data/labels_train_10knoise.npy')
btrain = np.load('/app/data/bit_signals_train_10knoise.npy')

xtest = np.load('/app/data/feature_vectors_test_10knoise.npy')
ytest = np.load('/app/data/labels_test_10knoise.npy')
btest = np.load('/app/data/bit_signals_test_10knoise.npy')

xvalidation = np.load('/app/data/feature_vectors_val_10knoise.npy')
yvalidation = np.load('/app/data/labels_val_10knoise.npy')
bvalidation = np.load('/app/data/bit_signals_val_10knoise.npy')


ffn = FFN(xtrain=xtrain,ytrain=ytrain,xtest=xtest,ytest=ytest,
          xvalidation=xvalidation,yvalidation=yvalidation)

start_time = time.time()

ffn.fit()

# print(time.time() - start_time)

print("score = ",ffn.evaluation())

ypred = ffn.ypred_train

optic_fiber_channel = Channel(number_symbols=32,N=2**10,T=200)

print(ypred.shape)

ypred.shape = (-1,2**10,2)

print(ypred.shape)

b_hat = []

print(ypred.shape)
for i in range(1):
    
    optic_fiber_channel.setter_function_nnet(ypred[i,:,0] + 1j * ypred[i,:,1])
    
    optic_fiber_channel.dmod()
    optic_fiber_channel.detector()
    optic_fiber_channel.symb_to_bit()
    
    print(optic_fiber_channel.s_hat)
    print(optic_fiber_channel.Constellation)
    
    temp = optic_fiber_channel.b_hat
    
    b_tilde = []
    
    for i in range(32):
        
        for j in range(4):
            
            b_tilde.append(int(temp[i][j]))

    b_hat.append(np.asarray(b_tilde))
    
BER = np.mean(np.abs(b_hat - btrain))

print(BER)