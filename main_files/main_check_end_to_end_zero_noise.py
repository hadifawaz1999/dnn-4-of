import os,sys

current_directory = os.path.dirname(os.path.realpath(__file__))
parent_directory = os.path.dirname(current_directory)
sys.path.append(parent_directory)

from cmath import log10
from math import log2

from channel import Channel

Bandwith = 10*1e9
L = 1000*1e3
T = 40
N = 2**10
p = 0.5
Nb = 32*4
M=16
Ns = Nb // int(log2(M))


optic_fiber_channel = Channel(Bandwith=Bandwith,T=T,N=N,p=p,number_symbols=Ns,nsp=1,Length=L)
optic_fiber_channel.channel(z=L)
optic_fiber_channel.equalize(z=L)
optic_fiber_channel.dmod()
optic_fiber_channel.detector()
optic_fiber_channel.symb_to_bit()

#print(optic_fiber_channel.qzt)

s_score , b_score = optic_fiber_channel.evaluate_results()

print(optic_fiber_channel.sigma2)
print(optic_fiber_channel.sigma02)

print('s score : ',s_score)
print('b score : ',b_score)

#print(optic_fiber_channel.data_generator.s)
#print(optic_fiber_channel.s_hat)

#print(optic_fiber_channel.sigma2)