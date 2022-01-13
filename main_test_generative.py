from pickletools import optimize
from data import DataGenerator
from nnet_gen import Generative_Model
from params import Params
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras

params = Params(N=2**10,nz=500,T=10,number_symbols=3)
dg = DataGenerator(N=2**10,T=10,number_symbols=3)



dg.source()
dg.bit_to_symb()
dg.mod()

s_label = dg.s

NNET_GEN = Generative_Model(params=params)

x = dg.q0t

x /= np.sqrt((np.sum(np.abs(x)**2)))

y = NNET_GEN.nnet_generator(x=x)

plt.figure(figsize=(20,10))

plt.figure(figsize=(20,10))

plt.xlabel(r'time')
plt.ylabel(r'| . |')
plt.plot(params.t,np.abs(x),color='black',label=r'$|q(t,0)|$',lw=3)
plt.plot(params.t,np.abs(y),color='red',label=r'$|q(t,L)|$',lw=3)

plt.legend()
plt.savefig('plots/q0t_vs_qLt_generaive_model.png')

# print(np.sum(np.abs(y)**2))

# calculate BER

y = np.asarray([[y[i].real , y[i].imag] for i in range(len(y))])

print(y.shape)

input_layer = keras.layers.Input(y.shape)
x = keras.layers.Flatten()(input_layer)
x = keras.layers.Dense(units=2**9,activation='tanh')(x)
x = keras.layers.Dense(units=2**7,activation='tanh')(x)

x = keras.layers.Dense(units=2**7,activation='tanh')(x)

x = keras.layers.Dense(units=2**8,activation='tanh')(x)
output_layer = keras.layers.Dense(units=dg.number_symbols*2,activation='linear')(x)

my_model = keras.models.Model(inputs=input_layer,outputs=output_layer)

my_loss = keras.losses.MeanSquaredError()

my_optimizer = keras.optimizers.SGD(lr=0.1)

my_model.compile(loss=my_loss,optimizer=my_optimizer)

my_model.summary()

keras.utils.plot_model(my_model,'my_model.png',show_shapes=True)

y.shape = (1,-1,2)
# y.shape = (1,-1)


s_label_1d = s_label.flatten()


history = my_model.fit(x=y,y=s_label_1d.reshape(1,-1),epochs=1000,batch_size=1,verbose=True)

loss = history.history['loss']

plt.clf()

plt.plot(loss,color='blue',label='training_loss',lw=3)
plt.legend()
plt.savefig('loss.png')

ypred = my_model.predict(y)

i = np.random.randint(low=0,high=100)

# print(ypred[0,i],ypred[0,dg.number_symbols+i])
# print(s_label[i])

print(np.asarray(ypred))
print(s_label)