import numpy as np
from math import *
from data import DataGenerator
import matplotlib.pyplot as plt

T = 4

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

my_data_generator = DataGenerator()

my_data_generator.source(N=nb,p=p)
b = my_data_generator.bernoulli

my_data_generator.bit_to_symb()
s = my_data_generator.s

print(s.shape)

my_data_generator.mod(t=t)

q0t_list = my_data_generator.q0t_list

q0t = my_data_generator.q0t

q0t_norm = [sqrt(q0t[i].real**2 + q0t[i].imag**2) for i in range(N)]

print(len(q0t_list))

for i in range(len(q0t_list)):
    print(i)
    q0t_list_i_norm = [sqrt(q0t_list[i,j].real**2 + q0t_list[i,j].imag**2) for j in range(N)]
    
    # plt.plot(q0t_list_i_norm,label='fction'+str(i))
    
plt.plot(q0t_norm,label='q0t')