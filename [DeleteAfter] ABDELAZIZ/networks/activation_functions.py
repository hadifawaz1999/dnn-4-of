import numpy as np

class Sigmoid():
    def __call__(self, x):
        return 1 / (1 + np.exp(-x))

    def gradient(self, x):
        return self.__call__(x) * (1 - self.__call__(x))

class Softmax():
    def __call__(self, x):
        e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e_x / np.sum(e_x, axis=-1, keepdims=True)

    def gradient(self, x):
        p = self.__call__(x)
        return p * (1 - p)

class TanH():
    def __call__(self, x):
        return 2 / (1 + np.exp(-2*x)) - 1

    def gradient(self, x):
        return 1 - np.power(self.__call__(x), 2)

class ReLU():
    def __call__(self, x):
        return np.where(x >= 0, x, 0)

    def gradient(self, x):
        return np.where(x >= 0, 1, 0)

class LeakyReLU():
    def __init__(self, alpha=0.2):
        self.alpha = alpha

    def __call__(self, x):
        return np.where(x >= 0, x, self.alpha * x)

    def gradient(self, x):
        return np.where(x >= 0, 1, self.alpha)

class ELU():
    def __init__(self, alpha=0.1):
        self.alpha = alpha 

    def __call__(self, x):
        return np.where(x >= 0.0, x, self.alpha * (np.exp(x) - 1))

    def gradient(self, x):
        return np.where(x >= 0.0, 1, self.__call__(x) + self.alpha)

class SELU():
    def __init__(self):
        self.alpha = 1.6732632423543772848170429916717
        self.scale = 1.0507009873554804934193349852946 

    def __call__(self, x):
        return self.scale * np.where(x >= 0.0, x, self.alpha*(np.exp(x)-1))

    def gradient(self, x):
        return self.scale * np.where(x >= 0.0, 1, self.alpha * np.exp(x))

class SoftPlus():
    def __call__(self, x):
        return np.log(1 + np.exp(x))

    def gradient(self, x):
        return 1 / (1 + np.exp(-x))


class PhaseActivation:

    def __call__(self, x):
        dz = 1/500
        abs2 = np.absolute(x)**2
        return x*np.exp(2*dz*abs2)

    def gradient(self, x):
        dz = 1/500
        #return np.abs(np.exp(2j*dz*(x)**2)) + np.abs(4j*((x)**2)*np.exp(dz*(x)**2))
        abs1 = np.absolute(x)
        abs2 = np.absolute(x)**2
        #grad = np.exp(2*dz*abs2) + x*4*abs1*np.exp(2*dz*abs2))
        grad = np.absolute(np.exp(2*dz*abs2)) + np.absolute(x*4*abs1*np.exp(2*dz*abs2))
        """
        print("abs1 = ",abs1)
        print("abs2 = ",abs2)
        print("grad = ",grad)
        """
        return grad