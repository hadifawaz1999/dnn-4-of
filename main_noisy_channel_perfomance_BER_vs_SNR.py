import numpy as np
import matplotlib.pyplot as plt
from channel import Channel

Bandwith = 1
L = 1e3
T = 50
N = 2**12

M_list = [2,4,8,16]

sigma2_list = np.arange(start=1e-1,stop=6e-1,step=5e-2)

number_of_symbols = 10**3

ber_list = []
snr_list = []

for M in M_list:
    
    optic_fiber_channel = Channel(Bandwith=Bandwith,N=N,T=T,Length=L,M=M,number_symbols=number_of_symbols)
    
    ber_list = []
    snr_list = []
    for sigma2 in sigma2_list:
        
        # snr = 10**(_snr/10)
        
        Constellation = optic_fiber_channel.constellation
        
        
        Es = np.mean(np.abs(Constellation)**2)
        
        
        
        optic_fiber_channel.setter_noise(sigma2=sigma2)
        
        optic_fiber_channel.channel()
        
        snr_list.append(Es / optic_fiber_channel.a)
        
        optic_fiber_channel.equalize()
        optic_fiber_channel.dmod()
        
        s_hat = optic_fiber_channel.s_hat
        
        s_hat_array = np.array([s_hat.real,s_hat.imag])
        
        s_hat_array.shape = (-1,2)
        
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.spines['left'].set_position('center')
        ax.spines['bottom'].set_position('center')
        ax.spines['right'].set_color('none')
        ax.spines['top'].set_color('none')

        plt.title(str(M)+'-QAM constellation with '+r'$\hat{s}$', pad=20)

        plt.scatter(s_hat_array[0,0],s_hat_array[0,1],color='blue',label=r'$\hat{s}$')
        plt.scatter(s_hat_array[1:,0],s_hat_array[1:,1],color='blue')
        
        test = 1

        for i in range(M):
            
            if test:
                plt.scatter(Constellation[i, 0],
                            Constellation[i, 1], color='red',label='Constellation')
                test = 0
            else:
                plt.scatter(Constellation[i, 0],
                            Constellation[i, 1], color='red')
                
            if Constellation[i, 1] > 0:
                plt.annotate(text=str(Constellation[i, 0])+' + j'+str(Constellation[i, 1]),
                             xy=(Constellation[i, 0],
                                 Constellation[i, 1]),
                             xytext=(Constellation[i, 0], Constellation[i, 1]+0.2))
            else:
                plt.annotate(text=str(Constellation[i, 0])+' - j'+str(abs(Constellation[i, 1])),
                             xy=(Constellation[i, 0], Constellation[i, 1]), ha='center', va='center',
                             xytext=(Constellation[i, 0], Constellation[i, 1]+0.2))
        
        
        
        plt.legend()
        plt.savefig('plots/question_19/s_hat_vs_const/'+str(M)+'_QAM_sigma2_'+str(sigma2)+'.png')
        plt.clf()
        
        optic_fiber_channel.detector()
        optic_fiber_channel.symb_to_bit()
        
        ser , ber = optic_fiber_channel.evaluate_results()
        
        ser , ber = 1 - ser , 1 - ber
        
        ber_list.append(ber)
    
    fig = plt.figure(figsize=(20,10))
    plt.plot(snr_list,ber_list)
    
    print(ber_list)
    print(snr_list)
    # plt.xscale('log')
    plt.yscale('log')
    plt.savefig('plots/question_19/ber_vs_snr/'+str(M)+'.png')
    plt.clf()