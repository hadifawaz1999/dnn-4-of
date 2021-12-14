import numpy as np
from numpy.core.defchararray import count
from data import DataGenerator
import matplotlib.pyplot as plt

# Test Draw Constellation

'''
points = [-3, -1, 1, 3]

const1 = np.asarray([[i, 3] for i in points])
const2 = np.asarray([[i, 1] for i in points])
const3 = np.asarray([[i, -1] for i in points])
const4 = np.asarray([[i, -3] for i in points])

Constellation = np.concatenate( ( np.concatenate( (const1, const2), axis=0),
                                        np.concatenate((const3, const4), axis=0) ),axis=0 )

print(Constellation)


fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.spines['left'].set_position('center')
ax.spines['bottom'].set_position('center')
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')

plt.title('16-QAM constellation', pad = 20)

for i in range(16):
    plt.scatter(Constellation[i,0],Constellation[i,1],color='red')
    if Constellation[i,1] > 0:
        plt.annotate(text=str(Constellation[i,0])+' + j'+str(Constellation[i,1]),
                     xy=(Constellation[i,0],Constellation[i,1]),
                     xytext=(Constellation[i,0],Constellation[i,1]+0.2))
    else:
        plt.annotate(text=str(Constellation[i,0])+' - j'+str(abs(Constellation[i,1])),
                     xy=(Constellation[i,0],Constellation[i,1]),ha='center',va='center',
                     xytext=(Constellation[i,0],Constellation[i,1]+0.2))
plt.show()

'''

# Test Bernoulli


dg = DataGenerator()

dg.source(N=16,p=0.5)

# a=dg.bernoulli
# print(a)
# print(sum(a[a==1]),1024-sum(a[a==1]))

dg.bit_to_symb()

print(dg.bernoulli)
s = dg.s
print(s)