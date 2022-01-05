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

fft_frequencies = optic_fiber_channel.fft_frequencies

q0t = A1 * np.exp(- (t+t0)**2 / (2 * D**2)) + A2 * np.exp(- (t-t0)**2 / (2 * D**2))


optic_fiber_channel.setter_function(q0t)

optic_fiber_channel.channel(z=1)

q0f = optic_fiber_channel.q0f

q1f = optic_fiber_channel.qzf

plt.figure(figsize=(20,10))
plt.title('Channel ISI simulation fequency domain without equalization z=1 '+r'$t_0=$'+str(t0))
plt.plot(fft_frequencies,np.abs(q0f),label="q(f,0)",color='blue',lw=3)
plt.plot(fft_frequencies,np.abs(q1f),label='q(f,1)',color='red',lw=3)
plt.legend()
plt.savefig('plots/ISI_simulation_freq_domain_without_equalization_q0t_q1t_t0='+str(t0)+'.png')