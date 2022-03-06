import os,sys

current_directory = os.path.dirname(os.path.realpath(__file__))
parent_directory = os.path.dirname(current_directory)
sys.path.append(parent_directory)

import numpy as np
from math import *
from data import DataGenerator
from channel import Channel
import matplotlib.pyplot as plt
import scipy

dg = DataGenerator(Bandwith=10,T=10,N=2**10,number_symbols=3)

dg.source()
dg.bit_to_symb()
dg.mod()
dg.plot_q0t()


# transmitter = Channel()

# z = 1
# transmitter.channel(z=z)

# qzf = transmitter.qzf

# qzt = transmitter.qzt
