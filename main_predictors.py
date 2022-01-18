import numpy as np
from FFN import FFN
from sklearn.model_selection import train_test_split
import time

feature_vectors = np.load(file='/app/data/feature_vectors.npy')
labels = np.load(file='/app/data/labels.npy')

xtrain , xtest , ytrain , ytest = train_test_split(feature_vectors,labels,test_size=0.20)

xtrain , xvalidation , ytrain , yvalidation = train_test_split(xtrain,ytrain,test_size=0.125)

ffn = FFN(xtrain=xtrain,ytrain=ytrain,xtest=xtest,ytest=ytest,
          xvalidation=xvalidation,yvalidation=yvalidation)

start_time = time.time()

ffn.fit()

print(time.time() - start_time)