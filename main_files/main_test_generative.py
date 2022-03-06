import os,sys

current_directory = os.path.dirname(os.path.realpath(__file__))
parent_directory = os.path.dirname(current_directory)
sys.path.append(parent_directory)

from data import DataGenerator
from nnet_gen import Generative_Model
from params import Params
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
import time

start_time = time.time()

T = 200
N = 2**10

params = Params(N=N,nz=500,T=T,number_symbols=32)
dg = DataGenerator(N=N,T=T,number_symbols=32)

dg.source()
dg.bit_to_symb()
dg.mod()

s_label = dg.s

NNET_GEN = Generative_Model(params=params)

x = dg.q0t

# x = np.exp(-params.t**2/2)

# x = 1 / np.cosh(params.t)

# x = np.sinc()

# x /= np.sqrt((np.sum(np.abs(x)**2)))

y = NNET_GEN.nnet_generator(x=x)

print(np.sum(np.abs(y)**2))

plt.figure(figsize=(20,10))

plt.xlabel(r'time')
plt.ylabel(r'| . |')
plt.plot(params.t,np.abs(x),color='black',label=r'$|q(t,0)|$',lw=7)
plt.plot(params.t,np.abs(y),color='red',label=r'$|q(t,L)|$',lw=3)

plt.legend()
plt.savefig('../plots/q0t_vs_qLt_generaive_model.png')

plt.clf()

q0f = np.fft.fftshift(np.fft.fft(x))

yf = np.fft.fftshift(np.fft.fft(y))

plt.figure(figsize=(20,10))

plt.xlabel(r'frequency')
plt.ylabel(r'| . |')
plt.plot(params.f,np.abs(q0f),color='black',label=r'$|q(t,0)|$',lw=7)
plt.plot(params.f,np.abs(yf),color='red',label=r'$|q(t,L)|$',lw=3)

plt.legend()
plt.savefig('../plots/q0f_vs_qLf_generaive_model.png')

print(np.sum(np.abs(y)**2))

# calculate BER

exit()

print("time : ",time.time() - start_time)

y = np.asarray([[y[i].real , y[i].imag] for i in range(len(y))])

print(y.shape)

input_layer = keras.layers.Input(y.shape)
x = keras.layers.Flatten()(input_layer)
x = keras.layers.Dense(units=2**9,activation='tanh')(x)
x = keras.layers.Dense(units=2**7,activation='tanh')(x)

x = keras.layers.Dense(units=2**7,activation='tanh')(x)

x = keras.layers.Dense(units=2**8,activation='tanh')(x)
# output_layer = keras.layers.Dense(units=dg.number_symbols*2,activation='linear')(x)
output_layer = keras.layers.Dense(units=N*2,activation='linear')(x)

my_model = keras.models.Model(inputs=input_layer,outputs=output_layer)

my_loss = keras.losses.MeanSquaredError()

my_optimizer = keras.optimizers.SGD(lr=0.5)

my_model.compile(loss=my_loss,optimizer=my_optimizer)

my_model.summary()

# keras.utils.plot_model(my_model,'my_model.png',show_shapes=True)

y.shape = (1,-1,2)
# y.shape = (1,-1)


s_label_1d = s_label.flatten()

q0t_flatten = np.asarray([[dg.q0t[i].real , dg.q0t[i].imag] for i in range(len(dg.q0t))]).flatten()


history = my_model.fit(x=y,y=q0t_flatten.reshape(1,-1),epochs=1000,batch_size=1,verbose=True)

loss = history.history['loss']


plt.plot(loss,color='blue',label='training_loss',lw=3)
plt.legend()
plt.savefig('loss.png')

ypred = my_model.predict(y)

print(ypred.shape)

ypred = np.asarray([complex(ypred[0,i],ypred[0,i+1]) for i in range(0,2*N,2)])


plt.clf()
plt.figure(figsize=(20,10))

plt.plot(dg.t,np.abs(dg.q0t),color="black",lw=5,label="q0t")
plt.plot(dg.t,np.abs(ypred),color='red',lw=3,label='q0tprime')
plt.legend()
plt.savefig('/app/comparaison.png')
exit()

i = np.random.randint(low=0,high=100)

# print(ypred[0,i],ypred[0,dg.number_symbols+i])
# print(s_label[i])

print(np.asarray(ypred))
print(s_label)