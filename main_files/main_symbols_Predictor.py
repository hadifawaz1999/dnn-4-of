import os,sys

current_directory = os.path.dirname(os.path.realpath(__file__))
parent_directory = os.path.dirname(current_directory)
sys.path.append(parent_directory)

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from custom_loss_function import constellation_loss_function
from Models_pred.CNN_symbols import CNN_symbols
from channel import Channel
from data import DataGenerator
import matplotlib.pyplot as plt

# feature_vectors = np.load(file='/app/data/feature_vectors_10knoise.npy')
# bit_signals = np.load(file='/app/data/bit_signals_10knoise.npy')
# labels = np.load(file='/app/data/labels_10knoise.npy')

# print(feature_vectors.shape)
# print(labels.shape)

gpu_path = '../'

xtrain = np.load(gpu_path+'data/feature_vectors_train_10knoise_SNR35_10.npy')
ytrain = np.load(gpu_path+'data/labels_train_10knoise_SNR35_10.npy')
btrain = np.load(gpu_path+'data/bit_signals_train_10knoise_SNR35_10.npy')
sbtrain=np.load(gpu_path+'data/symbols_train_10knoise_SNR35_10.npy')

xtest = np.load(gpu_path+'data/feature_vectors_test_10knoise_SNR35_10.npy')
ytest = np.load(gpu_path+'data/labels_test_10knoise_SNR35_10.npy')
btest = np.load(gpu_path+'data/bit_signals_test_10knoise_SNR35_10.npy')
sbtest=np.load(gpu_path+'data/symbols_test_10knoise_SNR35_10.npy')

xvalidation = np.load(gpu_path+'data/feature_vectors_val_10knoise_SNR35_10.npy')
yvalidation = np.load(gpu_path+'data/labels_val_10knoise_SNR35_10.npy')
bvalidation = np.load(gpu_path+'data/bit_signals_val_10knoise_SNR35_10.npy')
sbval=np.load(gpu_path+'data/symbols_val_10knoise_SNR35_10.npy')


sbtrain.shape = (-1,64)
sbtest.shape = (-1,64)
sbval.shape = (-1,64)

print(np.mean(xtrain[0,:,0]),np.std(xtrain[0,:,0]))


cnn = CNN_symbols(xtrain=xtrain,ytrain=sbtrain,xtest=xtest,ytest=sbtest,
          xvalidation=xvalidation,yvalidation=sbval)

#start_time = time.time()

cnn.fit()

# print(time.time() - start_time)

print("score = ",cnn.evaluation())

# ypred = cnn.ypred_train

ypred = cnn.ypred

optic_fiber_channel = Channel(number_symbols=32,N=2**10,T=40,Length=1000e3,Bandwith=10e9)

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
    
BER = np.mean(np.abs(b_hat - btest))

# BER = np.mean(np.abs(b_hat - btrain))


def hb_p(p):
    
    return -p*np.log2(p)-(1-p)*np.log2(1-p)

print("BER :" , BER)
print("Mutual Information :" , 1-hb_p(BER))