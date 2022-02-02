from pickletools import optimize
from turtle import shape
from warnings import filters
import numpy as np
import tensorflow as tf

gpu_path=''
xtrain = np.load(gpu_path+'data/feature_vectors_train_10knoise_L1000.npy')
xtest = np.load(gpu_path+'data/feature_vectors_test_10knoise_L1000.npy')
xvalidation = np.load(gpu_path+'data/feature_vectors_val_10knoise_L1000.npy')

input_layer = tf.keras.layers.Input(xtrain.shape[1:])
conv_1 = tf.keras.layers.Conv1D(filters=2,kernel_size=103,activation='relu')(input_layer)
conv_2 = tf.keras.layers.Conv1D(filters=2,kernel_size=206,activation='relu')(conv_1)
conv_3 =tf.keras.layers.Conv1D(filters=2,kernel_size=206,activation='relu')(conv_2)
conv_4 = tf.keras.layers.Conv1D(filters=2,kernel_size=1,activation='relu')(conv_3)
output_layer = tf.keras.layers.UpSampling1D(size = 2)(conv_4)

mymodel = tf.keras.models.Model(inputs=input_layer,outputs = output_layer)
mymodel.summary()

myoptimizer = tf.keras.optimizers.SGD(lr = 0.01)
mymodel.compile(loss = 'MAE',optimizer=myoptimizer)

mymodel.fit(xtrain,xtrain,batch_size=300,epochs=500)

xpred= mymodel.predict(xtrain)

print(np.mean(np.abs(xtrain-xpred)))


