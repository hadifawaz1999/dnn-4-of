import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from tqdm import tqdm

class ADAM(Optimizer) :

    #--------------------------------------------------------------------------------------------------------------#

    def __init__(self):
        super()

    #--------------------------------------------------------------------------------------------------------------#

    def ADAM(self, X_train, X_test, y_train, y_test, W1, W2, lambda1, lambda2, beta1, beta2, n_neuron, N, d, use_one_rs, with_print) :
    
        lossTrain = np.zeros(N)
        lossTest = np.zeros(N)
        
        # For W1
        U1 = np.zeros((n_neuron,d))
        V1 = np.zeros((n_neuron,d))
        
        # For W2
        U2 = np.zeros((1,n_neuron))
        V2 = np.zeros((1,n_neuron))
        
        # delta to prevent any division by zero
        delta = 1e-5
        
            
        if (with_print):
            
            for i in tqdm(range(N)) :
            
                # Compute Loss on prediction            
                lossTrain_ = self.computeLoss(X_train.T, y_train, W1, W2, lambda1, lambda2)
                lossTest_ = self.computeLoss(X_test.T, y_test, W1, W2, lambda1, lambda2)

                lossTrain[i] = lossTrain_
                lossTest[i] = lossTest_

                if (use_one_rs):

                    # compute gradient for one random sample i
                    random_sample_indice = np.random.randint(X_train.shape[0])        
                    random_X = X_train[random_sample_indice].reshape(-1,1)
                    random_y = y_train[random_sample_indice].reshape(-1,1)       

                    grad_1, grad_2 = self.computePartialDerivative(random_X, random_y, W1, W2, lambda1, lambda2)

                else :
                    # compute gradient on full X_train
                    grad_1, grad_2 = self.computePartialDerivative(X_train.T, y_train, W1, W2, lambda1, lambda2)

                # approximate bias for FoM
                U1 = beta1*U1 + (1-beta1)*grad_1
                U2 = beta1*U2 + (1-beta1)*grad_2

                # approximate bias for SoM
                V1 = beta2*V1 + (1-beta2)*grad_1*grad_1
                V2 = beta2*V2 + (1-beta2)*grad_2*grad_2


                # compute bias term of FoM
                x1 = 1- (beta1**i) + delta
                bias_U1 = U1/x1
                bias_U2 = U2/x1

                #compute bias term of SoM
                x2 = 1- (beta2**i) + delta
                bias_V1 = V1/x2
                bias_V2 = V2/x2

                # update weights
                inverse_1 = np.linalg.pinv((np.sqrt(bias_V1) + delta))
                inverse_2 = np.linalg.pinv((np.sqrt(bias_V2) + delta))

                W1 = W1 - inverse_1.T*bias_U1
                W2 = W2 - (inverse_2*bias_U2.T).T

                # to avoid the blowing weight effect
                W1 = self.minmax.fit_transform(W1)
                W2 = self.minmax.fit_transform(W2)
                
            MSE_train = self.MSE(lossTrain)
            MSE_test = self.MSE(lossTest)
            
            print("W2.shape : ", W2.shape)
            print("W1.shape : ", W1.shape)
            print("X.shape : ", X.shape)
            print("Test-Loss mean : ", lossTest.mean())
            print("MSE_test : ", MSE_test)
            
            return  W1, W2, lossTrain, lossTest, MSE_train, MSE_test
            
        else :
            for i in range(N) :
            
                # Compute Loss on prediction
                lossTrain_ = self.computeLoss(X_train.T, y_train, W1, W2, lambda1, lambda2)
                lossTest_ = self.computeLoss(X_test.T, y_test, W1, W2, lambda1, lambda2)

                lossTrain[i] = lossTrain_
                lossTest[i] = lossTest_  

                if (use_one_rs):

                    # compute gradient for one random sample i
                    random_sample_indice = np.random.randint(X_train.shape[0])        
                    random_X = X_train[random_sample_indice].reshape(-1,1)
                    random_y = y_train[random_sample_indice].reshape(-1,1)       

                    grad_1, grad_2 = self.computePartialDerivative(random_X, random_y, W1, W2, lambda1, lambda2)

                else :
                    # compute gradient for X_train
                    grad_1, grad_2 = self.computePartialDerivative(X_train.T, y_train, W1, W2, lambda1, lambda2)

                # approximate bias for FoM
                U1 = beta1*U1 + (1-beta1)*grad_1
                U2 = beta1*U2 + (1-beta1)*grad_2

                # approximate bias for SoM
                V1 = beta2*V1 + (1-beta2)*grad_1*grad_1
                V2 = beta2*V2 + (1-beta2)*grad_2*grad_2


                # compute bias term of FoM
                x1 = 1- (beta1**i) + delta
                bias_U1 = U1/x1
                bias_U2 = U2/x1

                #compute bias term of SoM
                x2 = 1- (beta2**i) + delta
                bias_V1 = V1/x2
                bias_V2 = V2/x2

                # update weights
                inverse_1 = np.linalg.pinv((np.sqrt(bias_V1) + delta))
                inverse_2 = np.linalg.pinv((np.sqrt(bias_V2) + delta))

                W1 = W1 - inverse_1.T*bias_U1
                W2 = W2 - (inverse_2*bias_U2.T).T

                # to avoid the blowing weight effect
                W1 = self.minmax.fit_transform(W1)
                W2 = self.minmax.fit_transform(W2)
                
            MSE_train = self.MSE(lossTrain)
            MSE_test = self.MSE(lossTest)
            
            return  W1, W2, lossTrain, lossTest, MSE_train, MSE_test

    #--------------------------------------------------------------------------------------------------------------#

    def KFoldCrossValidation_ADAM(self, df, k, X, Y, lambda1, lambda2, beta1, beta2, min_neurons, max_neurons, N, d, use_one_rs, with_print):
    
        d = X.shape[1]
        ts = int(len(X)/k)
        loss = []
        mse = []
        best_val = []
        with_print = 0
        neurons = range(min_neurons,max_neurons+1)
        
        #df = getPreprocessedData()
        
        
        for i in tqdm(range(k)) :
            # fix number of neurons
            for n_neuron in neurons:
                # fix beta 1
                for b1 in beta1 :
                    # fix beta 2
                    for b2 in beta2 :
                        # fix lambda 1
                        for l1 in lambda1 :
                            # fix lambda 2
                            for l2 in lambda2 :
                                # shuffle the dataset
                                df = shuffle(df)
                                # to avoid having same random_state in each iteration
                                randomValue = np.random.randint(50,size=1)[0]
                                # The test sample resprensents the N/k size of the dataset with KFold
                                X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=ts, random_state=randomValue)

                                W1 = np.random.rand(n_neuron,d)
                                W2 = np.random.rand(1,n_neuron)

                                # Training and testing
                                W1, W2, lossTrain, lossTest, MSE_train, MSE_test = ADAM(X_train, X_test, y_train, y_test, W1, W2, l1, l2, b1, b2, n_neuron, N, d, use_one_rs, with_print)
                                
                                # cost
                                real_loss = self.computeLoss(X_test.T, y_test, W1, W2, l1, l2)  
                                loss.append(real_loss)

                                loss_arr = np.array(loss)
                                real_mse = self.MSE(loss_arr)
                                mse.append(real_mse)
                            
                                # hyper params
                                best_val.append((l1,l2,b1,b2,n_neuron))
                    
        # return the best two params l1, l2
        
        plt.plot(loss)
        plt.title('Evolution of the loss by adding more neurons')
        plt.show()
        
        l1_, l2_, b1_, b2_ , n_neuron = best_val[loss.index(min(loss))]
        return loss, mse, best_val, l1_, l2_, b1_, b2_ , n_neuron

    #--------------------------------------------------------------------------------------------------------------#
    
    def getBestParams(self, lossTestingKFold, best_val):
   
        print ("len(lossTestingKFold) = ", len(lossTestingKFold))
        print ("len(best_val) = ", len(best_val))
        
        l1_, l2_, b1_, b2_ , n_neuron = best_val[lossTestingKFold.index(min(lossTestingKFold))]
        print("Best lambda1 : ", l1_)
        print("Best lambda2 : ", l2_)
        print("Best beta 1 : ", b1_)
        print("Best beta 2 : ", b2_)
        print("Best n_neuron : ", n_neuron)
        
        print("The loss we had with those three params : ", lossTestingKFold[lossTestingKFold.index(min(lossTestingKFold))])
        
        return l1_, l2_, b1_, b2_, n_neuron
    
    #--------------------------------------------------------------------------------------------------------------#
