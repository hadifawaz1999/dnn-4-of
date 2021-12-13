import numpy as np
import matplotlib.pyplot as plt
points = [-3, -1, 1, 3]

const1 = np.asarray([[i, 3] for i in points])
const2 = np.asarray([[i, 1] for i in points])
const3 = np.asarray([[i, -1] for i in points])
const4 = np.asarray([[i, -3] for i in points])

Constellation = np.concatenate( ( np.concatenate( (const1, const2), axis=0),
                                        np.concatenate((const3, const4), axis=0) ),axis=0 )

print(Constellation)


plt.scatter(Constellation[:,0],Constellation[:,1])
plt.show()