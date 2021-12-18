import numpy as np
from math import *
from data import DataGenerator
import matplotlib.pyplot as plt
import scipy

T = 10

N = 2**10

dt = T / N

t = np.arange(start=-T/2,stop=T/2,step=dt)

print(t.shape)

F = 1 / dt

df = 1 / T

f = np.arange(start=-F/2,stop=F/2,step=df)

print(f.shape)

M = 16

n = 3

nb = n * int(log2(M))

p = 0.5

my_Bandwith = 6

my_data_generator = DataGenerator(Bandwith=my_Bandwith)

my_data_generator.source(N=nb,p=p)
b = my_data_generator.bernoulli

my_data_generator.bit_to_symb()
s = my_data_generator.s

print(s.shape)

my_data_generator.mod(t=t)

q0t_list = my_data_generator.q0t_list

q0t = my_data_generator.q0t

q0t_norm = [sqrt(q0t[i].real**2 + q0t[i].imag**2) for i in range(N)]

plt.plot(t,q0t_norm,color='black')
plt.ylabel(r'|q(0,t)|')
plt.xlabel(r'time')
plt.title('Norm of '+r'q(0,t)'+' vs time with Bandwith = '+str(my_Bandwith))
plt.savefig('norm_q0t.png')
plt.clf()


# for i in range(len(q0t_list)):
    
#     q0t_i_FFT = scipy.fft.fft(q0t_list[i])
#     plt.plot(q0t_i_FFT)
    
q0t_FFT = scipy.fft.fft(q0t)    

# plt.plot(q0t,label='q(t,0)',color='orange')
plt.plot(f,q0t_FFT,color='red')
plt.xlabel(r'frequency')
plt.ylabel(r'FFT(q(0,t))')
plt.title(r'q(0,t)'+'in frequency domain with Bandwith = '+str(my_Bandwith))
plt.savefig('fft_q0t.png')