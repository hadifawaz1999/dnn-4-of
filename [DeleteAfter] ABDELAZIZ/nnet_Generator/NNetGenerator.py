import numpy as np
from scipy.linalg import dft
from tqdm import tqdm

class NNetGenerator :

    #-------------------------------------------------------------------------------------------------#

    def __init__(self, parameters):
        self.params = parameters

    #-------------------------------------------------------------------------------------------------#
    
    @DeprecationWarning
    def H(self, f, z):
        """ Compute the channel transfer function.
        @param f : frequency vector.
        @param z : scalar.
        """

        omega = 2*np.pi*f
        return np.exp(1j*z*omega**2)
       
    #-------------------------------------------------------------------------------------------------#

    def nnet_gen(self,  X):

      """ Simulate a NNet by computing the V matrix (V is Y in the implementation).
          @param X : data.
      """
      #X = X.reshape((-1, 1))
      X =  np.asarray(X)
      dz = self.params.z/self.params.nz


      for j in range(self.params.nz):

          # Linear transformation    
          X = np.fft.ifft(np.fft.fft(X)*np.exp(1j*dz*(self.params.w**2)))

          # Non-Linear transformation
          X = self.activation(X, dz)

          # Complex Gaussian Noise 
          Z = self.noise(len(X), self.params.sigma2 , self.params.B, dz)
        
          Y = X + Z
          X = Y
          
      # Output
      Y = np.squeeze(X)
      
      return Y

    #-------------------------------------------------------------------------------------------------#


    def activation(self, x, dz):
        """ Computes the activation of a neuron.
            @param X : data.
            @param dz : epsilon variation.
        """
        return x*np.exp(1j*dz*np.absolute(x)**2)

    #-------------------------------------------------------------------------------------------------#


    def noise(self, n, sigma2, B, dz):
        """Produces a Circular Gaussian Noise.
            @param n : number of samples, ie. size of the output vector.
            @param sigma2 : data.
            @param B : Bandwidth.
            @param dz : epsilon variation.
        """
        
        # Noise power
        Pn = sigma2 * B * dz

        return np.sqrt(Pn/2)*(np.random.randn(n) + 1j*np.random.randn(n))
        
    #-------------------------------------------------------------------------------------------------#



    def activation2(self, X, eps, gama):
        """ Computes the activation of a neuron.
            @param X : data.
        """
        norm = np.absolute(X)**2
        
        return X*np.exp(1j*gama*eps*norm)

    #------------------------------------------------------------------------------#





    def nnet_linear_gen(self, beta, power, eps, omegas, N, X, n_layers):
        """ Simulate the linear part of a NNet by computing the V matrix.
            @param N : Number of observations.
            @param X : data.
            @param n_layers : Number of layers.
        """
        
        np.set_printoptions(precision=2, suppress=True)  # for compact output
        
        W = np.fftshift(np.convolve(x,f))
        return np.fft([np.convolve( np.fft(x) , np.exp(complex(0,1)*to*(omegas)**2) )])
        

    def nnet_gen2(self, beta, power, eps, omegas, N, X, n_layers):
        """ Simulate a NNet by computing the V matrix.
            @param N : Number of observations.
            @param X : data.
            @param n_layers : Number of layers.
        """
    
        np.set_printoptions(precision=2, suppress=True)  # for compact output
        
        D = dft(N)
        R = np.diag([np.exp(complex(0,1)*(beta/2)*eps*(omegas[i])**2) for i in range(len(omegas))])
        W = np.dot(D.T, np.dot(R,D))
        
        # Deep Neural Network
        for j in range(n_layers):        
            V = np.dot(W, X)
            # Gaussian Noise
            Z = np.random.multivariate_normal(np.zeros(2), power*np.eye(2), size=N).view(np.complex128)   
            X = V + Z
                
        #print("Z : ", Z)
        print()
        print("V : ",V)
        
        #print("W : ",W)
        print()
        #print("R : ",R)
        return V