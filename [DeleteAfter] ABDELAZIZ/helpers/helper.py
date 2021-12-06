from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import datetime

import tensorflow as tf
from keras.callbacks import History 

import sys
sys.path.append("C:/Users/admin/Desktop/IP Paris/MICAS/Cours/910/913 - Deep Learning/Project/")

from helpers.processing import fromRealToComplex
from sklearn.metrics import mean_squared_error

from Channel.parameters import *
from Channel.transmitor import *
from Channel.modulator import *
from Channel.channel import *
from Channel.equalizer import *
from Channel.demodulator import *
from Channel.detector import *
from nnet_Generator.NNetGenerator import *


#-----------------------------------------------------------------------------------------#

# Generating the dataset
def generateDataset(nbrOfObservations, parameters, transmitor, modulator, nnetGen, demodulator, detector, isGaussian =False):

    X = np.ndarray((nbrOfObservations, parameters.N), dtype=np.complex128)
    y = np.ndarray((nbrOfObservations, parameters.N), dtype=np.complex128)    
    bits_in = np.ndarray((nbrOfObservations, parameters.nb))
    bhat_out = np.ndarray((nbrOfObservations, parameters.nb))
    symb_in = np.ndarray((nbrOfObservations, parameters.n))
    symb_out = np.ndarray((nbrOfObservations, parameters.n))


    print("[INFO] Generating the dataset...")

    for i in tqdm(range(nbrOfObservations)):
        
        # Constellation
        constellation = transmitor.build_constellations(parameters.M)
        
        #source
        source = transmitor.source(parameters.nb , parameters.p) # USE IT FOR NEXT PART OF BITS
        
        # Bits to Symboles - symbol sequence
        bitsToSymbols = transmitor.bit_to_symb(source, parameters.M)

        # symbol sequence, we create a list of values of the complex symbols to use them in modulation
        s = transmitor.bit_to_symb(source, parameters.M)
        
        # channel - we take gaussian input
        if isGaussian:
            q0t = parameters.A*np.exp(-parameters.t**2) 

        else :
            q0t = modulator.mod(parameters.t,s, parameters.B)  
        
        # Neural Net Generator
        y_gen = nnetGen.nnet_gen(q0t)
        # equalized y_gen
        qzte, qzfe = equalizer.equalize(parameters.t, y_gen, parameters.z) # equalized output
        qzte = qzte.reshape(1,-1)
        # demodulation
        shat = demodulator.demod(parameters.t, parameters.dt, qzte, parameters.B, parameters.n)

        # detection
        stilde, indexes = detector.detector(shat, parameters.M)
        bhat = detector.symbols_to_bit(indexes, parameters.M)
        
        # SOURCE IS THE SEQUENCE OF BITS TO BE LEARNED
        
        # Modulated signal
        X[i] = np.squeeze(q0t)
        # Neural Net Generator
        y[i] = y_gen
        # original bit sequence (source)
        bits_in[i] = source
        # estimated bit sequence
        bhat_out[i] = bhat
        # original symbol sequence (source)

        symb_in[i] = np.squeeze(s)
        # estimated symbol sequence
        symb_out[i] = np.squeeze(shat)
        
    print("[INFO] The dataset is ready now !")

    return X, y, bits_in, bhat_out, symb_in, symb_out

#-----------------------------------------------------------------------------------------#

    
def generate_dataset_multiple_power_symbols(nbrOfObservations, nsymbols_arr, M_arr, power_arr, isGaussian=False):
       
    # bandwidth
    bandwidth = 1
    # Sample size
    Nt = 2**10
    # Number of Layers of the Generative network
    nLayers = 500
    
    for power in power_arr :
        for M in M_arr :
            for nsymbols in nsymbols_arr :
                
                # Number of bits
                nb = int(nsymbols * np.log2(M)) 
                timeMesh = int( (nb/bandwidth)+ (10*2/ nb) )
                # Initialize parameters
                parameters = Parameters(bandwidth, nsymbols, M, Nt, nLayers, timeMesh)
                # Initialize the Transmitor
                transmitor = Transmitor()
                # Initialize the Transmitor
                modulator = Modulator()
                # Initialize the Channel
                channel = Channel()
                # Initialize the Equalizer
                equalizer = Equalizer()
                # Initialize the NNetGenerator
                nnetGen = NNetGenerator(parameters)
                # Initialize the Detector
                detector = Detector(transmitor)
                # Initialize the Demodulator
                demodulator = Demodulator()
                
                X, y, bits_in, bhat_out, symb_in, symb_out = generateDataset(nbrOfObservations, parameters, transmitor, modulator, equalizer, nnetGen, demodulator, detector, isGaussian)
                
                # saving the dataset
                file_saved_name = "../data/new/data_power_"+str(power)+"_nsymbols_"+str(nsymbols)+"_M_"+str(M)+".npz"
                np.savez_compressed(file_saved_name, X=X, y=y, bits_in=bits_in, bhat_out=bhat_out, symb_in=symb_in, symb_out=symb_out)

                print("[INFO] Saved generated data for power = ", power, " , nsymbols = ",nsymbols , " , M = ",M)

#-----------------------------------------------------------------------------------------#

# Generating the dataset
def generateDataset_old(nbrOfObservations, parameters, transmitor, modulator, nnetGen, isGaussian =False):

    X = np.ndarray((nbrOfObservations, parameters.N), dtype=np.complex128)
    y = np.ndarray((nbrOfObservations, parameters.N), dtype=np.complex128)

    print("[INFO] Generating the dataset...")

    for i in tqdm(range(nbrOfObservations)):
        
        # Constellation
        constellation = transmitor.build_constellations(parameters.M)
        
        #source
        source = transmitor.source(parameters.nb , parameters.p) # USE IT FOR NEXT PART OF BITS
        
        # Bits to Symboles - symbol sequence
        #bitsToSymbols = transmitor.bit_to_symb(source, constellation)

        # symbol sequence
        #mapping_dict, mapping_list, mapping_list_symbols_only = transmitor.bit_to_symb(source, constellation)
        # we create a list of values of the complex symbols to use them in modulation
        val_s = transmitor.bit_to_symb(source, parameters.M)
        

        # channel - we take gaussian input
        if isGaussian:
            q0t = parameters.A*np.exp(-parameters.t**2) 
            #q0f = np.fft.fft(q0t)

        else :
            q0t = modulator.mod(parameters.t,val_s, parameters.B)  
            #q0f = np.fft.fft(q0t)

        # SOURCE IS THE SEQUENCE OF BITS TO BE LEARNED
        X[i] = np.squeeze(q0t)

        # Neural Net Generator
        y[i] = nnetGen.nnet_gen(q0t)
        
            
    print("[INFO] The dataset is ready now !")

    return X, y

#-----------------------------------------------------------------------------------------#

# Generating the dataset
def generate_dataset_batch(nbrOfObservations, parameters,  transmitor, modulator, nnetGen, batch_size, index_start, isGaussian=False):


    X = np.ndarray((batch_size, parameters.N), dtype=np.complex128)
    y = np.ndarray((batch_size, parameters.N), dtype=np.complex128)

    print("[INFO] Generating the dataset...")

    #changed range from 0... not 1 to save last batch
    for i in tqdm(range(0,nbrOfObservations+1)):
        
        # we save data per batch to avoid missing data due to colab timeout...
        if i % batch_size == 0 and i != 0:
            file_saved_name = "../data/data_"+str(index_start)+".npz"
            np.savez_compressed(file_saved_name, X=X, y=y)
            X = np.ndarray((batch_size, parameters.N), dtype=np.complex128)
            y = np.ndarray((batch_size, parameters.N), dtype=np.complex128)
            print("[INFO] Saved generated data index ", index_start)
            index_start = index_start + 1

        # Constellation
        constellation = transmitor.build_constellations(parameters.M)
        
        #source
        source = transmitor.source(parameters.nb , parameters.p) # USE IT FOR NEXT PART OF BITS
        
        # Bits to Symboles - symbol sequence
        #bitsToSymbols = transmitor.bit_to_symb(source, constellation)

        # symbol sequence
        #mapping_dict, mapping_list, mapping_list_symbols_only = transmitor.bit_to_symb(source, constellation)
        # we create a list of values of the complex symbols to use them in modulation
        val_s = transmitor.bit_to_symb(source, parameters.M)
        

        # channel - we take gaussian input
        if isGaussian:
            q0t = parameters.A*np.exp(-parameters.t**2) 
            #q0f = np.fft.fft(q0t)

        else :
            q0t = modulator.mod(parameters.t,val_s, parameters.B)  
            #q0f = np.fft.fft(q0t)

        # SOURCE IS THE SEQUENCE OF BITS TO BE LEARNED
        index = i % batch_size
        X[index] = np.squeeze(q0t)

        # Neural Net Generator
        y[index] = nnetGen.nnet_gen(q0t)
        
            
    print("[INFO] The dataset is ready now !")
    return X, y

#-----------------------------------------------------------------------------------------#

@DeprecationWarning
def createDataFrame(X,y) :
    X_ = np.column_stack((X.real,X.imag))
    y_ = np.column_stack((y.real,y.imag))
    data = np.column_stack((X_,y_))

    df = pd.DataFrame(data)
    print("df.shape : ",df.shape)
    
    return df

#-----------------------------------------------------------------------------------------#

@DeprecationWarning
def test_dataset(X, parameters, channel, equalizer, index):

    """ Plots some visuals of a generated dataset (at index) """

    withPlot = True
    q0t = X[index] #fromRealToComplex(l)
    q0f = np.fft.fft(q0t)

    qzt, qzf = channel.channel(parameters.t, q0t, parameters.z, parameters.sigma02, parameters.B)

    # Equalization
    qzte, qzfe = equalizer.equalize(parameters.t, qzt, parameters.z)

    if withPlot :
        fig2, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(6, figsize = (15, 15))
        fig2.tight_layout(pad=7.0)

        ###
        ax1.plot(parameters.t, np.squeeze(q0t.real), 'b-')
        ax1.set_xlabel("Time")
        ax1.set_ylabel("Re(q0t)")
        ax1.set_title("Modulation - Time plot - Re(q0t)")

        ###
        ax2.plot(parameters.f, np.squeeze(np.abs(q0f)), 'b-')
        ax2.set_xlabel("Freq")
        ax2.set_ylabel("Abs(q0f)")
        ax2.set_title("Modulation - Freq plot - |q0f|")

        ###        
        ax3.plot(parameters.t, np.squeeze(qzt.real), 'b-')
        ax3.set_xlabel("Time")
        ax3.set_ylabel("Re(qzt)")
        ax3.set_title("Channel - Time plot - Re(qzt)")

        ###
        ax4.plot(parameters.f, np.squeeze(np.abs(qzf)), 'b-')
        ax4.set_xlabel("Freq")
        ax4.set_ylabel("Abs(qzf)")
        ax4.set_title("Channel - Freq plot - |qzf|")

        ###
        ax5.plot(parameters.t, np.squeeze(qzte.real), 'b-')
        ax5.set_xlabel("Time")
        ax5.set_ylabel("Abs(qzf)")
        ax5.set_title("Equalization - Time plot - Re(qzte)")

        ###
        ax6.plot(parameters.f, np.squeeze(np.abs(qzfe)), 'b-')
        ax6.set_xlabel("Freq")
        ax6.set_ylabel("Abs(qzf)")
        ax6.set_title("Equalization - Freq plot - |qzfe|")

        plt.show()
        
#-----------------------------------------------------------------------------------------#


def run(transmitor, modulator, channel, equalizer, parameters, isGaussian=False, s=2):

    """ runs all the functions to see the output of each module of the channel """

    # builds the constellation 
    constellation = transmitor.build_constellations(parameters.M) # 16-QAM
    
    # Bernoulli source, random bits sequence
    b = transmitor.source(parameters.nb,parameters.p)
    print("b : ",b)

    # symbol sequence, we create a list of values of the complex symbols to use them in modulation
    #val_s = transmitor.bit_to_symb(b, constellation)
    val_s = transmitor.bit_to_symb(b, parameters.M)

    # modulation   
    # channel - if gaussian input
    if isGaussian:
        print("isGaussian")
        q0t = parameters.A*np.exp(-parameters.t**2/s) 
        q0f = np.fft.fft(q0t)
        
    else :
        q0t = modulator.mod(parameters.t,val_s, parameters.B)  
        q0f = np.fft.fft(q0t)        
    
    print("q0t : ",q0t)
    print("q0f : ",q0f)


    # Pass through the channel
    qzt, qzf = channel.channel(parameters.t, q0t, parameters.z, parameters.sigma02, parameters.B)

    # Equalization
    qzte, qzfe = equalizer.equalize(parameters.t, qzt, parameters.z)

    fig2, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(6, figsize = (25, 25))
    fig2.tight_layout(pad=10.0)

    ###
    ax1.plot(parameters.t, np.squeeze(q0t.real), 'b-')
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Re(q0t)")
    ax1.set_title("Modulation - Time plot - Re(q0t)")

    ###
    ax2.plot(parameters.f, np.squeeze(np.abs(q0f)), 'b-')
    ax2.set_xlabel("Freq")
    ax2.set_ylabel("Abs(q0f)")
    ax2.set_title("Modulation - Freq plot - |q0f|")

    ###        
    ax3.plot(parameters.t, np.squeeze(qzt.real), 'b-')
    ax3.set_xlabel("Time")
    ax3.set_ylabel("Re(qzt)")
    ax3.set_title("Channel - Time plot - Re(qzt)")

    ###
    ax4.plot(parameters.f, np.squeeze(np.abs(qzf)), 'b-')
    ax4.set_xlabel("Freq")
    ax4.set_ylabel("Abs(qzf)")
    ax4.set_title("Channel - Freq plot - |qzf|")

    ###
    ax5.plot(parameters.t, np.squeeze(qzte.real), 'b-')
    ax5.set_xlabel("Time")
    ax5.set_ylabel("Abs(qzf)")
    ax5.set_title("Equalization - Time plot - Re(qzte)")

    ###
    ax6.plot(parameters.f, np.squeeze(np.abs(qzfe)), 'b-')
    ax6.set_xlabel("Freq")
    ax6.set_ylabel("Abs(qzf)")
    ax6.set_title("Equalization - Freq plot - |qzfe|")

    plt.show()

#-----------------------------------------------------------------------------------------#

def test(X, parameter, nnetGen, channel,equalizer):

    """ Tests the neural net generator w.r.t an input X of modulated signal, if
        X is none, then it creates a Gaussian vector 
    """

    # Signal
    if X is None :
        X = parameter.A*np.exp(-parameter.t**2/parameter.s)

    # NNet generator
    Y1 = nnetGen.nnet_gen(X)

    # Channel
    Y2, _ = channel.channel(parameter.t, X, parameter.z, 0, 0)

    # Equalizer
    Y2e, _ = equalizer.equalize(parameter.t, Y2, parameter.z)
    Y1e, _ = equalizer.equalize(parameter.t, np.squeeze(Y1), parameter.z)

    # Fourier Transform
    Xf = np.fft.fftshift(np.fft.fft(X))
    Yf = np.fft.fftshift(np.fft.fft(Y1))
    
    #Plots
    fig = plt.figure(figsize=(25, 25))
    
    # Input signal
    ax = fig.add_subplot(711)
    fig.suptitle('X -- Y1 -- Y2')
    ax.set_xlabel('Time')
    ax.set_ylabel('X')
    ax.plot(parameter.t, X, 'g--')
    ax.axhline(y=0, color='black')
    ax.axvline(x=0, color='black')

    # NNet generator
    ax = fig.add_subplot(712)
    ax.set_xlabel('Time')
    ax.set_ylabel('Y1')
    ax.plot(parameter.t, np.squeeze(np.absolute(Y1)), 'r--')
    ax.axhline(y=0, color='black')
    ax.axvline(x=0, color='black')
    
    # Signal after channel
    ax = fig.add_subplot(713)
    ax.set_xlabel('Time')
    ax.set_ylabel('Y2')
    ax.plot(parameter.t, np.absolute(Y2), 'b--')
    ax.axhline(y=0, color='black')
    ax.axvline(x=0, color='black')
    
    # NNet gen signal after equalization 
    ax = fig.add_subplot(714)
    ax.set_xlabel('Time')
    ax.set_ylabel('Y1 equalized')
    ax.plot(parameter.t, np.absolute(Y1e), 'b--')
    ax.axhline(y=0, color='black')
    ax.axvline(x=0, color='black')
    
    # NNet gen signal after equalization
    ax = fig.add_subplot(715)
    ax.set_xlabel('Time')
    ax.set_ylabel('Y2 equalized')
    ax.plot(parameter.t, np.absolute(Y2e), 'b--')
    ax.axhline(y=0, color='black')
    ax.axvline(x=0, color='black')
    
    # Input vs output
    
    # in time
    ax = fig.add_subplot(716)
    ax.set_xlabel('Time')
    ax.plot(parameter.t, X, 'b--', label='input')
    ax.plot(parameter.t, np.absolute(Y2), 'r--', label='output')
    ax.axhline(y=0, color='black')
    ax.axvline(x=0, color='black')
    
    # in freq
    ax = fig.add_subplot(717)
    ax.set_xlabel('Frequency')
    ax.plot(parameter.f, np.absolute(Xf), 'b--', label='input')
    ax.plot(parameter.f, np.absolute(Yf), 'r--', label='output')
    ax.axhline(y=0, color='black')
    ax.axvline(x=0, color='black')

    plt.show()
    
#-----------------------------------------------------------------------------------------#

def test_generated(X, y, index, parameter, channel, equalizer):

    """ Tests a generated y (at index) w.r.t an input X (at index) of modulated signal """

    # Signal : X
    X = np.squeeze(X[index])
    
    # NNet generator : y
    y_gen = np.squeeze(y[index])
    y = np.squeeze(y[index])

    w_ = np.fft.fftshift(2*np.pi*parameter.f)
    w_ = np.reshape(-1,1)
    y_vf = np.fft.ifft(np.exp(1j*parameter.z*(w_**2))*np.fft.fft(X))
    
    # Channel
    Y2, _ = channel.channel(parameter.t, X, parameter.z, parameter.sigma2, parameter.B )#0, 0)

    # Equalizer
    Y2e, _ = equalizer.equalize(parameter.t, Y2, parameter.z)
    Y1e, _ = equalizer.equalize(parameter.t, np.squeeze(y), parameter.z)

    # Fourier Transform
    Xf = np.fft.fftshift(np.fft.fft(X))
    Yf = np.fft.fftshift(np.fft.fft(y))
    
    #Plots
    
    fig, (ax1, ax2, ax3, ax4, ax5, ax6, ax7) = plt.subplots(7, figsize = (25, 25))
    #fig = plt.figure(figsize=(25, 25))
    fig.tight_layout(pad=3.0)
    # Input signal
    #ax = fig.add_subplot(711)
    fig.suptitle('Testing the generated data')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('X')
    ax1.set_title("Input Signal")
    ax1.plot(parameter.t, np.abs(X), 'g--')
    ax1.axhline(y=0, color='black')
    ax1.axvline(x=0, color='black')

    # NNet generator
    #ax = fig.add_subplot(712)
    ax2.set_xlabel('Time')
    ax2.set_ylabel('y')
    ax2.set_title("Output of the nnet gen")
    ax2.plot(parameter.t, np.abs(y_gen), 'r--')
    #ax2.plot(parameter.t, np.abs(y_vf), 'g--')
    ax2.axhline(y=0, color='black')
    ax2.axvline(x=0, color='black')
    
    # Signal after channel
    #ax = fig.add_subplot(713)
    ax3.set_xlabel('Time')
    ax3.set_ylabel('Y2')
    ax3.set_title("Output of X passing th channel function")
    ax3.plot(parameter.t, np.abs(Y2), 'b--')
    ax3.axhline(y=0, color='black')
    ax3.axvline(x=0, color='black')
    
    # NNet gen signal after equalization
    #ax = fig.add_subplot(714)
    ax4.set_xlabel('Time')
    ax4.set_ylabel('y equalized')
    ax4.set_title("Equalized output of the nnet gen")
    ax4.plot(parameter.t, np.abs(Y1e), 'b--')
    ax4.axhline(y=0, color='black')
    ax4.axvline(x=0, color='black')
    
    # NNet gen signal after equalization
    #ax = fig.add_subplot(715)
    ax5.set_xlabel('Time')
    ax5.set_ylabel('Y2 equalized')
    ax5.set_title("Equalized X passing th channel function")
    ax5.plot(parameter.t, np.abs(Y2e), 'b--')
    ax5.axhline(y=0, color='black')
    ax5.axvline(x=0, color='black')
    
    # Input vs output
    
    # in time
    #ax = fig.add_subplot(716)
    ax6.set_xlabel('Time')
    ax6.plot(parameter.t, np.abs(X), 'b--', label='input')
    ax6.plot(parameter.t, np.abs(Y2), 'r--', label='output')
    ax6.set_title("Input X vs X passing the channel function")
    ax6.axhline(y=0, color='black')
    ax6.axvline(x=0, color='black')
    ax6.legend()
    
    # in freq
    #ax = fig.add_subplot(717)
    ax7.set_xlabel('Frequency')
    ax7.plot(parameter.f, np.abs(Xf), 'b--', label='input')
    ax7.plot(parameter.f, np.abs(Yf), 'r--', label='output')
    ax7.set_title("Input X vs output of nnet gen (in Freq)")
    ax7.axhline(y=0, color='black')
    ax7.axvline(x=0, color='black')
    ax7.legend()

    plt.show()
    
#-----------------------------------------------------------------------------------------#

def train(model, X_train, X_test, y_train, y_test, Nepochs, batchSize, earlystopping_patience=5) :

    """ Trains a tensorflow model """

    # Fiting the model
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    earlystopping = tf.keras.callbacks.EarlyStopping(monitor ="val_loss",  
                                            mode ="min", patience = earlystopping_patience,  
                                            restore_best_weights = True) 
    
    history = History()

    model.fit( X_train, y_train, 
                batch_size = batchSize, 
                epochs = Nepochs, 
                verbose = 1, 
                validation_data =(X_test, y_test), 
                #callbacks =[history, earlystopping, tensorboard_callback])
                callbacks =[history, earlystopping, tensorboard_callback])
    
    return model

#-----------------------------------------------------------------------------------------------------#

def test_model(model, X_test, y_test, history, index_pred, with_error_plots, all_plots=False) :

    """ Test a tensorflow model """

    with_eval = False
    y_pred = evaluate(model, X_test, y_test, history, index_pred, with_eval, with_error_plots, all_plots)
    return y_pred

#-----------------------------------------------------------------------------------------------------#


def evaluate(model, X_test, y_test, history, parameters, channel, equalizer, index_pred, with_eval, with_error_plots, all_plots=False):

    """ Plots some visuals of the original and predicted data to compare them """
    # evaluating and printing results 
    if with_eval :
        score = model.evaluate(X_test, y_test, verbose = 0) 
        print('Test loss:', score[0]) 
        print('Test accuracy:', score[1]) 

    # Plots
    
    if with_error_plots : 
        plt.plot(history.history['mse'])
        plt.title("mse")
        plt.show()
        plt.plot(history.history['mae'])
        plt.title("mae")
        plt.show()
        plt.plot(history.history['cosine_proximity'])
        plt.title("cosine_proximity")
        plt.show()
        
    # prediction on one element
    X = X_test[index_pred]
    y_pred = model.predict(X)
    print("X_test.shape : ",X_test.shape)
    print("y_test.shape : ",y_test.shape)
    print("X.shape : ",X.shape)
    print("y_pred.shape : ",y_pred.shape)
    print("\n")

    X = X.reshape(-1,1)
    y_pred = y_pred.reshape(-1,1)
    print("After reshaping...")
    print("X.shape : ",X.shape)
    print("y_pred.shape : ",y_pred.shape)

    if all_plots:
    
        X_im = fromRealToComplex(X.T)
        y_im = fromRealToComplex(y_pred.T)

        y_qzt, y_qzf = channel.channel(parameters.t, X_im, parameters.z, parameters.sigma02, parameters.B)
        y_gen_equalized_t, y_gen_equalized_f = equalizer.equalize(parameters.t, y_im, parameters.z)
        y_qzt_equalized_t, y_qzt_equalized_f = equalizer.equalize(parameters.t, y_qzt, parameters.z)
        allPlots(y_qzt, y_qzf, y_qzt_equalized_t, y_qzt_equalized_f, y_im, y_gen_equalized_t, y_gen_equalized_f)

    return y_pred

#-----------------------------------------------------------------------------------------------------#

@DeprecationWarning
def allPlots(parameters, y_qzt, y_qzf, y_qzt_equalized_t, y_qzt_equalized_f, y_pred, y_pred_equalized_t, y_pred_equalized_f):
    
    """ Plots some visuals of the equalized original and predicted data to compare them """

    fig2, (ax1, ax2, ax3, ax4, ax5, ax6, ax7) = plt.subplots(7, figsize = (15, 15))
    fig2.tight_layout(pad=7.0)


    ###
    ax1.plot(parameters.t, np.squeeze(y_qzt.real), 'b-')
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Abs(y_qzt)")
    ax1.set_title("y_test passed in Channel function - Time plot - Re(y_qzt)")

    ###
    ax2.plot(parameters.f, np.squeeze(np.abs(y_qzf)), 'b-')
    ax2.set_xlabel("Freq")
    ax2.set_ylabel("Abs(y_qzf)")
    ax2.set_title("Equalization of y_test- Freq plot - |y_qzf|")

    ###
    ax3.plot(parameters.t, np.squeeze(np.abs(y_qzt_equalized_t)), 'b-')
    ax3.set_xlabel("Time")
    ax3.set_ylabel("Abs(y_qzt_equalized_t)")
    ax3.set_title("Equalization of y_test input in the Channel - Time plot - |y_qzt_equalized_t|")

    ###
    ax4.plot(parameters.f, np.squeeze(np.abs(y_qzt_equalized_f)), 'b-')
    ax4.set_xlabel("Freq")
    ax4.set_ylabel("Abs(y_qzt_equalized_f)")
    ax4.set_title("Equalization of y_test input in the Channel - Freq plot - |y_qzt_equalized_f|")

    ###
    ax5.plot(parameters.t, np.squeeze(np.absolute(y_pred)), 'b-')
    ax5.set_xlabel("Time")
    ax5.set_ylabel("Abs(y_pred)")
    ax5.set_title("Prediction (y_pred) - Time plot - |y_pred|")

    ###
    ax6.plot(parameters.t, np.squeeze(np.abs(y_pred_equalized_t)), 'b-')
    ax6.set_xlabel("Time")
    ax6.set_ylabel("Abs(y_pred_equalized_t)")
    ax6.set_title("Equalization of prediction (y_pred) - Time plot - |y_pred_equalized_t|")

    ###        
    ax7.plot(parameters.f, np.squeeze(np.abs(y_pred_equalized_f)), 'b-')
    ax7.set_xlabel("Freq")
    ax7.set_ylabel("Abs(y_pred_equalized_t)")
    ax7.set_title("Equalization of prediction (y_pred) - Freq plot - |y_pred_equalized_f|")

    plt.show()

#-----------------------------------------------------------------------------------------------------#

def plotResults(X, y_pred, parameters):

    """ Plots results in frequency domain """

    y_pred_im = fromRealToComplex(y_pred.T).T
    X = fromRealToComplex(X.T).T

    Xf = np.fft.fftshift(X)
    Yf = np.fft.fftshift(y_pred_im)

    fig3, (ax5) = plt.subplots(1, figsize = (10, 10))
    fig3.tight_layout(pad=7.0)

    ###
    ax5.plot(parameters.f, np.squeeze(np.absolute(Xf)), 'b-', label='nnet')
    ax5.set_xlabel("Frequency")
    ax5.set_ylabel("Abs(y_gen)")
    ax5.set_title("NNetGen - Time plot - |y_gen|")

    ###
    ax5.plot(parameters.f, np.squeeze(np.absolute(Yf)), 'r-', label='nnet')
    ax5.set_xlabel("Frequency")
    ax5.set_ylabel("Abs(y_gen)")
    ax5.set_title("NNetGen - Freq plot - |y_gen|")

#-----------------------------------------------------------------------------------------------------#

def test_prediction(y_test, y_pred, index_pred, parameter):

    """ Plots results in time and frequency domain """

    print("y_pred : ",y_pred)
    print("y_pred.shape : ",y_pred.shape)

    # Original
    y_ = y_test[index_pred].reshape(1,-1,1)
    y_test_im = fromRealToComplex(y_)
    # Predicted
    y_pred = y_pred.reshape(1,-1,1)
    y_pred_im = fromRealToComplex(y_pred)
    
    y_test_im = np.squeeze(y_test_im)
    y_pred_im = np.squeeze(y_pred_im)

    diff = y_ - y_pred
    diff_im = fromRealToComplex(diff)

    y_ = np.squeeze(y_)
    y_pred = np.squeeze(y_pred)
    diff_im = np.squeeze(diff_im)

    mse_diff = mean_squared_error(y_, y_pred)
    #mse_diff = (1/len(diff))*np.sum((diff**2))

    
    print("\n")
    print("y_test : ", y_)
    print("\n")

    print("y_pred : ", y_pred)
    print("\n")

    print("diff : ", diff)
    print("mse test : ", mse_diff)
    print("\n")

    # Equalizer
    #Y2e, _ = equalizer.equalize(parameter.t, y_test, parameter.z)
    #Y1e, _ = equalizer.equalize(parameter.t, y_pred, parameter.z)

    #Plots

    plt.plot(np.abs(y_pred))
    plt.title("predicted")
    plt.show()

    plt.plot(np.abs(y_test[index_pred]))
    plt.title("test")
    plt.show()
    fig = plt.figure(figsize=(15, 15))
    
    # y_test vs y_pred
    ax = fig.add_subplot(211)
    #ax.set_xlabel('Time')
    ax.plot(y_test[index_pred], 'b--', label='test')
    ax.plot(y_pred, 'r--', label='predicted')
    # ax.plot(parameter.t, y_test_im, 'b--', label='test')
    # ax.plot(parameter.t, y_pred_im, 'r--', label='predicted')
    # ax.plot(parameter.t, diff_im.T, 'g--', label='difference')
    ax.axhline(y=0, color='black')
    ax.axvline(x=0, color='black')
    ax.legend()

    # Fourier Transform
    Xf = np.fft.fftshift(np.fft.fft(y_test_im))
    Yf = np.fft.fftshift(np.fft.fft(y_pred_im))
    
    plt.xlabel('Frequency')
    plt.plot(parameter.f, np.abs(Xf), 'b--', label='input')
    plt.plot(parameter.f, np.abs(Yf), 'r--', label='output')
    plt.title("Input X vs output of nnet gen (in Freq)")
    plt.show()

    plt.plot(np.squeeze(abs(y_test[index_pred])), 'g--')
    plt.plot(np.squeeze(abs(y_pred)), 'b--')
    plt.show()

    plt.plot(np.squeeze(abs(y_test_im)), 'g--')
    plt.plot(np.squeeze(abs(y_pred_im)), 'b--')
    plt.show()

#-----------------------------------------------------------------------------------------#

def test_predicted(y_test, y_pred, index, parameter):

    """ Plots results in time domain """

    #X = X_test[index]

    print("y_pred : ",y_pred)
    print("y_pred.shape : ",y_pred.shape)

    # Original
    y_original = y_test[index].reshape(1,-1,1)
    y_test_im = fromRealToComplex(y_original)

    # Predicted
    y_predicted = y_pred.reshape(1,-1,1)
    y_pred_im = fromRealToComplex(y_predicted)
    
    # Getting rid of one dim
    y_test_im = np.squeeze(y_test_im)
    y_pred_im = np.squeeze(y_pred_im)

    # Compute the difference between the test and predicted data
    diff = y_original - y_predicted
    diff_im = fromRealToComplex(diff)

    y_original = np.squeeze(y_original)
    y_predicted = np.squeeze(y_predicted)
    diff_im = np.squeeze(diff_im)

    #Plots
    
    fig, (ax1) = plt.subplots(1, figsize = (25, 25))
    fig.tight_layout(pad=3.0)

    # Input signal
    fig.suptitle('Testing the predicted data')

    ax1.set_xlabel('Time')
    ax1.plot(parameter.t, np.abs(y_original), 'b--', label='y_test')
    ax1.plot(parameter.t, np.abs(y_predicted), 'r--', label='y_predicted')
    ax1.set_title("Input signal vs predicted signal.")
    ax1.axhline(y=0, color='black')
    ax1.axvline(x=0, color='black')
    ax1.legend()
    
    plt.show()
        
#-----------------------------------------------------------------------------------------#

def test_ypred(y_pred, parameter):
    
    """ Plots results in time domain """

    # Predicted 
    y_pred = y_pred.reshape(1,-1,1)
    y_pred_im = fromRealToComplex(y_pred)
    
    y_pred_im = np.squeeze(y_pred_im)

    y_pred = np.squeeze(y_pred)

    print("y_pred_im.shape : ",y_pred_im.shape)
    print("y_pred.shape : ",y_pred.shape)
    print("\n")

    # Equalizer
    #Y2e, _ = equalizer.equalize(parameter.t, y_test, parameter.z)
    #Y1e, _ = equalizer.equalize(parameter.t, y_pred, parameter.z)

    #Plots
    fig = plt.figure(figsize=(15, 15))
    
    # y_test vs y_pred
    ax = fig.add_subplot(211)
    ax.set_xlabel('Time')
    ax.plot(parameter.t, y_pred_im.T, 'r--', label='predicted')
    ax.axhline(y=0, color='black')
    ax.axvline(x=0, color='black')
    ax.legend()

    ax = fig.add_subplot(212)
    ax.set_xlabel('Time')
    ax.plot(y_pred, 'r--', label='predicted')
    ax.axhline(y=0, color='black')
    ax.axvline(x=0, color='black')

    plt.show()
    
#-----------------------------------------------------------------------------------------#

def evaluate_model(model, X_test, y_test, index_pred, opti, train_err, val_err):
    """ Test a trained model.
        @param model        : an instence of the NeuralNetwork class
        @param opti         : the used optimzer name in str format. used in the plot
        @param train_err    : Training error array
        @param opti         : Validation error array
    """
    # Training and validation error plot
    n = len(train_err)
    training, = plt.plot(range(n), train_err, label="Training Error")
    validation, = plt.plot(range(n), val_err, label="Validation Error")
    plt.legend(handles=[training, validation])
    plt.title("Error Plot using "+str(opti))
    plt.ylabel('Error')
    plt.xlabel('Iterations')
    plt.show()
    
    y_pred = model.predict(X_test[index_pred])
    
    plt.title("Predicted Signal")
    plt.plot(np.squeeze(np.abs(y_pred)))
    plt.show()
    
    plt.title("Predicted Signal vs Original")
    plt.plot(np.squeeze(np.abs(y_pred)), label="prediction")
    plt.plot(np.squeeze(np.abs(y_test[index_pred])), label="original")
    plt.legend()
    plt.show()
    
    return y_pred

#-----------------------------------------------------------------------------------------#