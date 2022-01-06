from math import log2
from channel import Channel

Bandwith = 6
L = 1e3
T = 50
N = 2**12
p = 0.5
Nb = 64
M=16
Ns = Nb // int(log2(M))


optic_fiber_channel = Channel(Bandwith=Bandwith,T=T,N=N,p=p,number_symbols=Ns,nsp=0)
optic_fiber_channel.channel(z=L)
optic_fiber_channel.equalize(z=L)
optic_fiber_channel.dmod()
optic_fiber_channel.detector()
optic_fiber_channel.symb_to_bit()

s_score , b_score = optic_fiber_channel.evaluate_results()

print('s score : ',s_score)
print('b score : ',b_score)