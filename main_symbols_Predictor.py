import numpy as np
from FFN import FFN
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import time
from channel import Channel
from data import DataGenerator
import matplotlib.pyplot as plt
from CNN_symbols import CNN_symbols

# feature_vectors = np.load(file='/app/data/feature_vectors_10knoise.npy')
# bit_signals = np.load(file='/app/data/bit_signals_10knoise.npy')
# labels = np.load(file='/app/data/labels_10knoise.npy')

# print(feature_vectors.shape)
# print(labels.shape)

xtrain = np.load('data/feature_vectors_train_10knoise.npy')
ytrain = np.load('data/labels_train_10knoise.npy')
btrain = np.load('data/bit_signals_train_10knoise.npy')
sbtrain=np.load('data/symbols_train_10knoise.npy')

xtest = np.load('data/feature_vectors_test_10knoise.npy')
ytest = np.load('data/labels_test_10knoise.npy')
btest = np.load('data/bit_signals_test_10knoise.npy')
sbtest=np.load('data/symbols_test_10knoise.npy')

xvalidation = np.load('data/feature_vectors_val_10knoise.npy')
yvalidation = np.load('data/labels_val_10knoise.npy')
bvalidation = np.load('data/bit_signals_val_10knoise.npy')
sbval=np.load('data/symbols_val_10knoise.npy')

sbtrain.shape = (-1,64)
sbtest.shape = (-1,64)
sbval.shape = (-1,64)


cnn = CNN_symbols(xtrain=xtrain,ytrain=sbtrain,xtest=xtest,ytest=sbtest,
          xvalidation=xvalidation,yvalidation=sbval)

start_time = time.time()

cnn.fit()

# print(time.time() - start_time)

print("score = ",cnn.evaluation())

ypred = cnn.ypred

optic_fiber_channel = Channel(number_symbols=32,N=2**10,T=200,Length=30)

print(ypred.shape)

ypred.shape = (-1,32,2)

print(ypred.shape)

b_hat = []

print(ypred.shape)
for i in range(len(ypred)):
    
    optic_fiber_channel.setter_function_s_hat(ypred[i,:,0] + 1j * ypred[i,:,1])
    optic_fiber_channel.detector()
    optic_fiber_channel.symb_to_bit()

    
    temp = optic_fiber_channel.b_hat
    
    b_tilde = []
    
    for i in range(32):
        
        for j in range(4):
            
            b_tilde.append(int(temp[i][j]))

    b_hat.append(np.asarray(b_tilde))
    
BER = np.mean(np.abs(b_hat - bvalidation))

print(BER)