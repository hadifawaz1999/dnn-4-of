import numpy as np
import matplotlib.pyplot as plt
from channel import Channel

M_list = [2,4,8,16]

SNR_list = np.arange(start=-4,stop=10,step=1)

for M in M_list:
    
    optic_fiber_channel = Channel(Bandwith=6,M=M,number_symbols=1e4,)