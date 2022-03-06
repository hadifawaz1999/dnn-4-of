import os,sys

current_directory = os.path.dirname(os.path.realpath(__file__))
parent_directory = os.path.dirname(current_directory)
sys.path.append(parent_directory)

from channel import Channel
import matplotlib.pyplot as plt
import numpy as np
import scipy

optic_fiber_channel = Channel(Bandwith=0,T=50,N=2**10)

A = 1

t = optic_fiber_channel.t
dt = optic_fiber_channel.data_generator.dt

f = optic_fiber_channel.data_generator.fft_frequencies


q0t = A * np.exp(-t**2)

q0f = np.fft.fft(q0t)


fig, ax = plt.subplots(nrows=1,ncols=2,figsize=(20,10))

ax[0].set_title('input time domain')
ax[0].plot(t,q0t,label='q(t,0)',color='red',lw=3)
ax[0].set_xlabel('time')
ax[0].set_ylabel('q(t,0)')

ax[0].legend()

ax[1].set_title('input frequency domain')
ax[1].plot(f,np.abs(q0f.real),label='q(f,0)',color='red',lw=3)
ax[1].set_xlabel('frequency')
ax[1].set_ylabel('|q(f,0)|')

ax[1].legend()
plt.savefig('../plots/input_on_t_input_on_f.png')

plt.clf()

optic_fiber_channel.setter_function(q0t)

L = 1e3

optic_fiber_channel.channel(z=L)

qlf = optic_fiber_channel.qzf

qlt = optic_fiber_channel.qzt


fig, ax = plt.subplots(nrows=1,ncols=2,figsize=(20,10))

ax[0].set_title('input time domain')
ax[0].plot(t,q0t,label='q(t,0)',color='red',lw=3)
ax[0].plot(t,qlt,label='q(t,L)',color='blue',lw=3)
ax[0].set_xlabel('time')

ax[0].legend()

ax[1].set_title('input frequency domain')
ax[1].plot(f,np.abs(q0f.real),label='q(f,0)',color='red',lw=3)
ax[1].plot(f,np.abs(qlf.real),label='q(f,L)',color='blue',lw=3)
ax[1].set_xlabel('frequency')

ax[1].legend()
plt.savefig('../plots/input_output_in_t_input_output_in_f.png')
plt.clf()


optic_fiber_channel.equalize(z=L)
qlfe = optic_fiber_channel.qzfe
qlte = optic_fiber_channel.qzte


plt.figure(figsize=(20,10))
plt.plot(t,q0t,label='input',color='red',lw=3)
plt.plot(t,qlte,label='output equalizer',color='blue',lw=3)
plt.legend()
plt.savefig('../plots/input_in_t_vs_output_equalizer_in_t.png')

plt.clf()

print(optic_fiber_channel.compare())


optic_fiber_channel.dmod()

shat = optic_fiber_channel.shat

print(shat.shape)
print(shat)