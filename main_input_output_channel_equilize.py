from channel import Channel
import matplotlib.pyplot as plt
import numpy as np
import scipy

optic_fiber_channel = Channel(Bandwith=0,T=10,N=2**7)

A = 1

t = optic_fiber_channel.t
f = optic_fiber_channel.data_generator.f

q0t = A * np.sinc(t)

q0f = scipy.fftpack.fft.fft(q0t)


fig, ax = plt.subplots(nrows=1,ncols=2,figsize=(20,10))

ax[0].set_title('input time domain')
ax[0].plot(t,q0t,label='q(t,0)',color='red',lw=3)
ax[0].set_xlabel('time')
ax[0].set_ylabel('q(t,0)')

ax[0].spines['left'].set_position('center')
ax[0].spines['right'].set_color('none')
ax[0].spines['top'].set_color('none')

ax[1].set_title('input frequency domain')
ax[1].plot(f,q0f.real,label='q(f,0)',color='red',lw=3)
ax[1].set_xlabel('frequency')
ax[1].set_ylabel('|q(f,0)|')
ax[1].spines['left'].set_position('center')
ax[1].spines['right'].set_color('none')
ax[1].spines['top'].set_color('none')

plt.legend()
plt.savefig('input_on_t_input_on_f.png')