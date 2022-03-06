import numpy as np
import matplotlib.pyplot as plt



gpu_path = ""
xtrain = np.load(gpu_path+'data/feature_vectors_train_10knoise_SNR35.npy')
ytrain = np.load(gpu_path+'data/labels_train_10knoise_SNR35.npy')
btrain = np.load(gpu_path+'data/bit_signals_train_10knoise_SNR35.npy')
sbtrain=np.load(gpu_path+'data/symbols_train_10knoise_SNR35.npy')

print(ytrain.shape)




plt.plot(ytrain[0])
plt.legend()
plt.show()