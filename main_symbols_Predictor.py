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

gpu_path = ''

xtrain = np.load(gpu_path+'data/feature_vectors_train_10knoise_SNR35.npy')
ytrain = np.load(gpu_path+'data/labels_train_10knoise_SNR35.npy')
btrain = np.load(gpu_path+'data/bit_signals_train_10knoise_SNR35.npy')
sbtrain=np.load(gpu_path+'data/symbols_train_10knoise_SNR35.npy')

xtest = np.load(gpu_path+'data/feature_vectors_test_10knoise_SNR35.npy')
ytest = np.load(gpu_path+'data/labels_test_10knoise_SNR35.npy')
btest = np.load(gpu_path+'data/bit_signals_test_10knoise_SNR35.npy')
sbtest=np.load(gpu_path+'data/symbols_test_10knoise_SNR35.npy')

xvalidation = np.load(gpu_path+'data/feature_vectors_val_10knoise_SNR35.npy')
yvalidation = np.load(gpu_path+'data/labels_val_10knoise_SNR35.npy')
bvalidation = np.load(gpu_path+'data/bit_signals_val_10knoise_SNR35.npy')
sbval=np.load(gpu_path+'data/symbols_val_10knoise_SNR35.npy')


sbtrain.shape = (-1,64)
sbtest.shape = (-1,64)
sbval.shape = (-1,64)

#xtrain[:,:,0] = (xtrain[:,:,0] - np.mean(xtrain[:,:,0],axis=1,keepdims=True)) / np.std(xtrain[:,:,0],axis=1,keepdims=True)
#xvalidation[:,:,0] = (xvalidation[:,:,0] - np.mean(xvalidation[:,:,0],axis=1,keepdims=True)) / np.std(xvalidation[:,:,0],axis=1,keepdims=True)
#xtest[:,:,0] = (xtest[:,:,0] - np.mean(xtest[:,:,0],axis=1,keepdims=True)) / np.std(xtest[:,:,0],axis=1,keepdims=True)
#xtrain[:,:,1] = (xtrain[:,:,1] - np.mean(xtrain[:,:,1],axis=1,keepdims=True)) / np.std(xtrain[:,:,1],axis=1,keepdims=True)
#xvalidation[:,:,1] = (xvalidation[:,:,1] - np.mean(xvalidation[:,:,1],axis=1,keepdims=True)) / np.std(xvalidation[:,:,1],axis=1,keepdims=True)
#xtest[:,:,1] = (xtest[:,:,1] - np.mean(xtest[:,:,1],axis=1,keepdims=True)) / np.std(xtest[:,:,1],axis=1,keepdims=True)

print(np.mean(xtrain[0,:,0]),np.std(xtrain[0,:,0]))


cnn = CNN_symbols(xtrain=xtrain,ytrain=sbtrain,xtest=xtest,ytest=sbtest,
          xvalidation=xvalidation,yvalidation=sbval)

start_time = time.time()

cnn.fit()

# print(time.time() - start_time)

print("score = ",cnn.evaluation())

# ypred = cnn.ypred_train

ypred = cnn.ypred

optic_fiber_channel = Channel(number_symbols=32,N=2**10,T=200,Length=30)

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

# BER = np.mean(np.abs(b_hat - btrain))


print(sbval)
print(ypred)

print(BER)