import os,sys

current_directory = os.path.dirname(os.path.realpath(__file__))
parent_directory = os.path.dirname(current_directory)
sys.path.append(parent_directory)

from channel import Channel
import matplotlib.pyplot as plt
import numpy as np

bandwith = 6
T = 50
N = 2**12
A1 = 1
A2 = 2
D = 1
t0 = 4

optic_fiber_channel = Channel(Bandwith=bandwith,T=T,N=N,nsp=0)

t = optic_fiber_channel.t

q0t = A1 * np.exp(- (t+t0)**2 / (2 * D**2)) + A2 * np.exp(- (t-t0)**2 / (2 * D**2))

optic_fiber_channel.setter_function(q0t)

optic_fiber_channel.channel(z=1)

q1t = optic_fiber_channel.qzt

plt.figure(figsize=(20,10))
plt.title('Channel ISI simulation without equalization z=1 '+r'$t_0=$'+str(t0))
plt.plot(t,q0t,label="q(t,0)",color='blue',lw=3)
plt.plot(t,q1t,label='q(t,1)',color='red',lw=3)
plt.legend()
plt.savefig('../plots/ISI_simulation_without_equalization_q0t_q1t_t0='+str(t0)+'.png')