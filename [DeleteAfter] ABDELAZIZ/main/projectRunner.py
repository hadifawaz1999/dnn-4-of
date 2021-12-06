import numpy as np
import matplotlib.pyplot as plt

class projectRunner :

    def __init__(self, parameters, transmitor, modulator, channel):
        self.params = parameters
        self.transmitor = transmitor
        self.modulator = modulator
        self.channel = channel

    #-------------------------------------------------------------------------------------------------#
    

    def argand(self, X):
        """Plots a complex constellation."""
        for i in range(len(X)):
            plt.plot([0,X[i].real],[0,X[i].imag],'ro-',label='python')
        limit=np.max(np.ceil(np.absolute(X))) # set limits for axis
        plt.xlim((-limit,limit))
        plt.ylim((-limit,limit))
        plt.ylabel('Imaginary')
        plt.xlabel('Real')
        plt.show()
    
    #-------------------------------------------------------------------------------------------------#

    def run(self):

        # builds the constellation 
        constellation = self.transmitor.build_constellations(self.params.M) # 16-QAM
    
        # plots the constellation 
        #self.argand(constellation)
        
        # Bernoulli source, random bits sequence
        b = self.transmitor.source(self.params.N,self.params.p)

        # symbol sequence
        s = self.transmitor.bit_to_symb(b, constellation)

        # modulation
        q0t = self.modulator.mod(self.params.t,constellation, self.params.B)  
        q0f = 1/q0t
        #channel
        qzt, qzf = self.channel.channel(self.params.t, q0t, self.params.z, self.params.sigma2, self.params.B)
        
        withPlot = 1
        if withPlot :
            fig, (ax1, ax2, ax3, ax4) = plt.subplots(4)


            ax1.plot(self.params.t, q0t)
            ax1.set_title("q0t")
            ####
            ax2.plot(self.params.f, np.abs(q0f))
            ax2.set_title("q0f")
            ####
            ax3.plot(self.params.t, qzt)
            ax3.set_title("qzt")
            ###
            ax4.plot(self.params.f, np.abs(qzf))
            ax4.set_title("qzf")
            plt.show()
