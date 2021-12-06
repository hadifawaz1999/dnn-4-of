
from __future__ import print_function
from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append("C:/Users/admin/Desktop/IP Paris/MICAS/Cours/910/913 - Deep Learning/Project/")

# Import helper functions
from networks import NeuralNetwork
from optimization.optimizers import StochasticGradientDescent, Adam, RMSprop, Adagrad, Adadelta
from networks.loss_functions import CrossEntropy, SquareLoss
from utils.misc import bar_widgets
from networks.layers import Dense, Dropout, Activation, BatchNormalization, Conv2D


import pandas as pd
from numpy import save
import matplotlib.pyplot as plt
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler

import time

#----------------------------------------------------------------------------#

def fromComplexToReal(vect_im):
  
   # vect_real = []
    N = vect_im.shape[0]
    M = vect_im.shape[1]*2
    vect_real = np.zeros((N,M))

    print("\n [INFO] {fromComplexToReal} runing...")
    print("\n [INFO] vect_real.shape : " , vect_real.shape)

    for i in tqdm(range(len(vect_im))) :
        tmp = []
        for j in range(len(vect_im[i])) :
            z = vect_im[i][j]
            tmp.append(z.real)
            tmp.append(z.imag)
        vect_real[i] = tmp
        #vect_real.append(tmp)

    #vect_real = np.asarray(vect_real)
    print(" [INFO] {fromComplexToReal} vect_real.shape : ",vect_real.shape)

    return vect_real

#----------------------------------------------------------------------------#

def fromRealToComplex(vect_real):
  
    # init variables
    #vect_im = []
    N = vect_real.shape[0]
    M = vect_real.shape[1]//2
    
    vect_im = np.zeros((N,M), dtype=np.complex64)  

    print("\n [INFO] {fromRealToComplex} runing...")
    # iterate over the vector
    for i in tqdm(range(len(vect_real))) :
        tmp = []
        j = 0

        # we can't use for loop with range() we have no control over the index
        while j < len(vect_real[i]) :

          # y = a+j*b
          im = vect_real[i][j] + 1j*vect_real[i][j+1]
          # we already processed two elements 
          j = j + 2
          # append the new element to the output array
          tmp.append(im)

        # matrix i*j
        #vect_im.append(tmp)
        vect_im[i] = tmp
    # to numpy
    #vect_im = np.asarray(vect_im)
    print("[INFO] {fromRealToComplex} vect_im.shape : ",vect_im.shape)

    return vect_im

#----------------------------------------------------------------------------#

def fromRealToComplex2(vect_real):
  
    # init variables
    vect_im = []

    print("\n [INFO] {fromRealToComplex} runing...")
    # iterate over the vector
    for i in tqdm(range(len(vect_real))) :
        tmp = []
        j = 0

        # we can't use for loop with range() we have no control over the index
        while j < len(vect_real[i]) :

          # y = a+j*b
          im = vect_real[i][j] + 1j*vect_real[i][j+1]
          # we already processed two elements 
          j = j + 2
          # append the new element to the output array
          tmp.append(im)

        # matrix i*j
        vect_im.append(tmp)
    # to numpy
    vect_im = np.asarray(vect_im)
    print("[INFO] {fromRealToComplex} vect_im.shape : ",vect_im.shape)

    return vect_im

#----------------------------------------------------------------------------#

def prepareDataFrame(X, y, scaling) :

    X_real = fromComplexToReal(X)
    y_real = fromComplexToReal(y)

    data = np.column_stack((X_real,y_real))

    df_ = pd.DataFrame(data)

    if scaling :
      scaler = MinMaxScaler()
      df_ = scaler.fit_transform(df_)
      
    df_ = shuffle(df_)

    print("{prepareDataFrame} df_.shape : ",df_.shape)

    return df_

#----------------------------------------------------------------------------#

def prepareTrainAndTestData(df, withReshape, ts):

  #df = df.set_index('Attribute',inplace=True)
  
    N_samples = df.shape[0]
    N_features = df.shape[1]//2
    N_cols = df.shape[1]

    if type(df) != np.ndarray :
        print("[INFO] - {prepareTrainAndTestData} converting from pandas to numpy...")
        df_np = df.to_numpy()
        y = df_np[:N_samples,0:N_features]
        X = df_np[:N_samples,N_features:N_cols]
    else :
        y = df[:N_samples,0:N_features]
        X = df[:N_samples,N_features:N_cols]

    N_features = len(X[0])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=ts)

    N_samples_train = len(X_train)
    N_samples_test = len(X_test)

    if withReshape :
        X_train = X_train.reshape(N_samples_train, N_features, 1)
        X_test = X_test.reshape(N_samples_test, N_features, 1)
        y_train = y_train.reshape(N_samples_train, N_features, 1)
        y_test = y_test.reshape(N_samples_test, N_features, 1)

    print(X.shape)
    print(y.shape)
    print(X_train.shape)
    print(y_train.shape)
    print(X_test.shape)
    print(y_test.shape)

    return X_train, X_test, y_train, y_test
  
#----------------------------------------------------------------------------#

def prepareDataFrame(X, y, scaling) :

    X_real = fromComplexToReal(X)
    y_real = fromComplexToReal(y)

    data = np.column_stack((X_real,y_real))

    df_ = pd.DataFrame(data)

    if scaling :
      scaler = MinMaxScaler()
      df_ = scaler.fit_transform(df_)
      
    df_ = shuffle(df_)

    print("{prepareDataFrame} df_.shape : ",df_.shape)

    return df_

#----------------------------------------------------------------------------#

def prepareTrainAndTestData(df, withReshape, ts):

  #df = df.set_index('Attribute',inplace=True)
  
    N_samples = df.shape[0]
    N_features = df.shape[1]//2
    N_cols = df.shape[1]

    if type(df) != np.ndarray :
        print("[INFO] - {prepareTrainAndTestData} converting from pandas to numpy...")
        df_np = df.to_numpy()
        y = df_np[:N_samples,0:N_features]
        X = df_np[:N_samples,N_features:N_cols]
    else :
        y = df[:N_samples,0:N_features]
        X = df[:N_samples,N_features:N_cols]

    N_features = len(X[0])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=ts)

    N_samples_train = len(X_train)
    N_samples_test = len(X_test)

    if withReshape :
        X_train = X_train.reshape(N_samples_train, N_features, 1)
        X_test = X_test.reshape(N_samples_test, N_features, 1)
        y_train = y_train.reshape(N_samples_train, N_features, 1)
        y_test = y_test.reshape(N_samples_test, N_features, 1)

    print(X.shape)
    print(y.shape)
    print(X_train.shape)
    print(y_train.shape)
    print(X_test.shape)
    print(y_test.shape)

    return X_train, X_test, y_train, y_test
  
#----------------------------------------------------------------------------#

def main(num_epochs, batch_size):

    #optimizer = Adam()
    optimizer = Adam()
    opti = "Adam"

    #-----
    # MLP
    #-----
    
    """
    PATH_DATA = '../data/data_90K.csv.gz'
    t1 = time.time()
    df = data = pd.read_csv(PATH_DATA, compression='gzip')
    t2 = time.time()
    print("Toral time to read the dataframe is : ", t2-t1, " seconds.")

    """
    t1 = time.time()
    #d2= np.load('../data/data_1_30K_P6e-2.npz')
    d2= np.load('../data/data_1_90K_P6e-2_P5e-2_P2.6e-2.npz')
    X = d2['X']
    y = d2['y']

    # if scaling=True, we are using min-max scaler
    df = prepareDataFrame(X, y, scaling=False)
    #df.to_csv("../data/data_90K.csv.gz", index=False, compression="gzip")
    t2 = time.time()
    print("Toral time to prepare the dataframe is : ", t2-t1, " seconds.")

    X = df.iloc[:, 2048:4096]
    y = df.iloc[:, 0:2048]

    n_samples, n_features = X.shape
    n_hidden = 512

    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, seed=1)
    withReshape = True
    ts = 0.4
    X_train, X_test, y_train, y_test = prepareTrainAndTestData(df, withReshape, ts)

    X_train = np.squeeze(X_train)
    X_test = np.squeeze(X_test)
    y_train = np.squeeze(y_train)
    y_test = np.squeeze(y_test)


    clf = NeuralNetwork(optimizer=optimizer,
                        loss=SquareLoss,
                        validation_data=(X_test, y_test))

    """
    clf.add(Dense(n_hidden, input_shape=(n_features,)))
    clf.add(Activation('leaky_relu'))
    clf.add(Dense(n_hidden))
    clf.add(Activation('leaky_relu'))
    clf.add(Dropout(0.25))
    clf.add(Dense(n_hidden))
    clf.add(Activation('leaky_relu'))
    clf.add(Dropout(0.25))
    clf.add(Dense(n_hidden))
    clf.add(Activation('leaky_relu'))
    clf.add(Dropout(0.25))
    clf.add(Dense(2048))
    clf.add(Activation('softmax'))
    """
    clf.add(Conv2D(n_filters=16, filter_shape=(3,3), stride=1, input_shape=(1,8,8), padding='same'))
    clf.add(Activation('selu'))

    clf.add(Dense(128, input_shape=(n_features,)))
    clf.add(Activation('selu'))
    clf.add(Dropout(0.25))
    clf.add(BatchNormalization())
    clf.add(Dense(64))
    clf.add(Activation('selu'))
    clf.add(Dropout(0.25))
    clf.add(BatchNormalization())
    clf.add(Dense(32))
    clf.add(Activation('selu'))
    clf.add(Dropout(0.25))
    clf.add(BatchNormalization())
    clf.add(Dense(64))
    clf.add(Activation('selu'))
    clf.add(Dropout(0.25))
    clf.add(BatchNormalization())
    clf.add(Dense(128))
    clf.add(Activation('selu'))
    clf.add(Dropout(0.25))
    clf.add(BatchNormalization())
    clf.add(Dense(n_features))
    clf.add(Activation('selu'))

    print ()
    clf.summary(name="MLP")
    
    print("[INFO] Batch size : ", batch_size)
    print("[INFO] Training epochs number : ", num_epochs)
    train_err, val_err = clf.fit(X_train, y_train, num_epochs, batch_size=batch_size)
    
    # Training and validation error plot
    n = len(train_err)
    training, = plt.plot(range(n), train_err, label="Training Error")
    validation, = plt.plot(range(n), val_err, label="Validation Error")
    plt.legend(handles=[training, validation])
    plt.title("Error Plot using "+str(opti))
    plt.ylabel('Error')
    plt.xlabel('Iterations')
    plt.show()

    _, loss = clf.test_on_batch(X_test, y_test)
    print ("Test loss:", loss)

    y_pred = clf.predict(X_test[0])
    print(y_pred)
    plt.title("Predicted Signal")
    plt.plot(np.squeeze(np.abs(y_pred)))
    plt.show()

    plt.title("Predicted Signal vs Original")
    plt.plot(np.squeeze(np.abs(y_pred)), label="prediction")
    plt.plot(np.squeeze(np.abs(y_test[0])), label="original")
    plt.legend()
    plt.show()

    print ("max val :", np.max(y_pred))
    print ("shape  :", y_pred.shape)


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_epochs", default=25, help="Number of training epochs.")
    parser.add_argument("--batch_size", default=128, help="Training batch size.")
    args = parser.parse_args()
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    main(num_epochs, batch_size)