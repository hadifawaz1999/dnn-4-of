import numpy as np
from numpy.fft import fft,ifft,fftshift
import matplotlib.pyplot as plt

T = 500
N=2**11
dt=T/N
z=1
t = np.arange(start=-T/2,stop=T/2,step=dt)

F=1/dt
df=1/T
f = np.arange(start=-F/2,stop=F/2,step=df)
w = 2*np.pi*f 
x = np.exp(-t**2/5)


h = np.exp(1j*w**2*z)
y = ifft(h*fftshift(fft(x)))

print(np.linalg.norm(x,ord=2),np.linalg.norm(y,ord=2))

plt.plot(t,np.abs(x))
plt.plot(t,np.abs(y))
plt.show()



plt.clf()

xf = fftshift(fft(x))
yf = fftshift(fft(y))


plt.plot(f,np.abs(xf))
plt.plot(f,np.abs(yf))
plt.show()