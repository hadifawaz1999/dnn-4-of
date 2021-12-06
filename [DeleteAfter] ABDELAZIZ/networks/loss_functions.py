from __future__ import division
import numpy as np

import sys
sys.path.append("C:/Users/admin/Desktop/IP Paris/MICAS/Cours/910/913 - Deep Learning/Project/")

from utils import accuracy_score
from networks.activation_functions import Sigmoid

class Loss(object):

    """ Parent class """
    
    def loss(self, y_true, y_pred):
        return NotImplementedError()

    def gradient(self, y, y_pred):
        raise NotImplementedError()

    def acc(self, y, y_pred):
        return 0

class SquareLoss(Loss):

    """ Compensate the problem of MSE in this case. """

    def __init__(self): pass

    def loss(self, y, y_pred):
        return 0.5 * np.power((y - y_pred), 2)

    def gradient(self, y, y_pred):
        return -(y - y_pred)

class MSE(Loss):

    """ Not good to train in our project, the vector y has a len of 2048, and the mean value is 0.04. 
        The loss will be so small and the network won't learn from it !
    """

    def __init__(self): pass

    def loss(self, y, y_pred):
        return (1/len(y))* np.power((y - y_pred), 2)

    def gradient(self, y, y_pred):
        return -(2/len(y))*(y - y_pred)

class CrossEntropy(Loss):

    """ Used in the bits-to-bits model """

    def __init__(self): pass

    def loss(self, y, p):
        # Avoid division by zero
        p = np.clip(p, 1e-15, 1 - 1e-15)
        return - y * np.log(p) - (1 - y) * np.log(1 - p)

    def acc(self, y, p):
        return accuracy_score(np.argmax(y, axis=1), np.argmax(p, axis=1))

    def gradient(self, y, p):
        # Avoid division by zero
        p = np.clip(p, 1e-15, 1 - 1e-15)
        return - (y / p) + (1 - y) / (1 - p)