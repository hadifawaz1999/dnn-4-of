import os,sys

current_directory = os.path.dirname(os.path.realpath(__file__))
parent_directory = os.path.dirname(current_directory)
sys.path.append(parent_directory)

import numpy as np
from Models_pred.FFN import FFN
from sklearn.model_selection import train_test_split
import time
from channel import Channel
from data import DataGenerator
import matplotlib.pyplot as plt
from Models_pred.CNN import CNN



gpu_path='../'


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


ffn = CNN(xtrain=xtrain,ytrain=ytrain,xtest=xtest,ytest=ytest,
          xvalidation=xvalidation,yvalidation=yvalidation)

start_time = time.time()

ffn.fit()

# print(time.time() - start_time)

print("score = ",ffn.evaluation())

ypred = ffn.ypred

optic_fiber_channel = Channel(number_symbols=32,N=2**10,T=40,Length=1000e3,Bandwith=10e9)

print(ypred.shape)

ypred.shape = (-1,2**10,2)

print(ypred.shape)

b_hat = []

print(ypred.shape)
for i in range(len(ypred)):
    
    optic_fiber_channel.setter_function_nnet(ypred[i,:,0] + 1j * ypred[i,:,1])
    
    optic_fiber_channel.dmod()
    optic_fiber_channel.detector()
    optic_fiber_channel.symb_to_bit()

    
    temp = optic_fiber_channel.b_hat
    
    b_tilde = []
    
    for i in range(32):
        
        for j in range(4):
            
            b_tilde.append(int(temp[i][j]))

    b_hat.append(np.asarray(b_tilde))
    
BER = np.mean(np.abs(b_hat - btest))

def hb_p(p):
    
    return -p*np.log2(p)-(1-p)*np.log2(1-p)

MI= 1-hb_p(BER)

print("BER :", BER)
print("Mutual Information", MI)