import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tqdm import tqdm

class Optimizer :

    #--------------------------------------------------------------------------------------------------------------#

    def __init__(self):
        super()
        self.minmax = MinMaxScaler()


    #--------------------------------------------------------------------------------------------------------------#

    def backPropagation(self, X_train, X_test, y_train, y_test, W1, W2, lambda1, lambda2, alpha, N, with_print) :
    
        lossTrain = np.zeros(N)
        lossTest = np.zeros(N)

        # just to avoid prints when we do cross validation   
        if (with_print):
            for i in tqdm(range(N)) :

                lossTrain_ = self.computeLoss(X_train, y_train, W1, W2, lambda1, lambda2)
                lossTest_ = self.computeLoss(X_test, y_test, W1, W2, lambda1, lambda2)

                lossTrain[i] = lossTrain_
                lossTest[i] = lossTest_

                grad_1, grad_2 = self.computePartialDerivative(X_train, y_train, W1, W2, lambda1, lambda2)

                W1 = W1 - alpha*grad_1
                W2 = W2 - alpha*grad_2
            
            MSE_train = self.MSE(lossTrain)
            MSE_test = self.MSE(lossTest)

            print("W2.shape : ", W2.shape)
            print("W1.shape : ", W1.shape)
            print("X.shape : ", X.shape)
            print("Test-Loss mean : ", lossTest.mean())
            print("MSE_test : ", MSE_test)
            
            return W1, W2, lossTrain, lossTest, MSE_train, MSE_test


        else :
            
            for i in range(N) :

                lossTrain_ = self.computeLoss(X_train, y_train, W1, W2, lambda1, lambda2)
                lossTest_ = self.computeLoss(X_test, y_test, W1, W2, lambda1, lambda2)

                lossTrain[i] = lossTrain_
                lossTest[i] = lossTest_

                grad_1, grad_2 = self.computePartialDerivative(X_train, y_train, W1, W2, lambda1, lambda2)

                W1 = W1 - alpha*grad_1
                W2 = W2 - alpha*grad_2
            
            MSE_train = self.MSE(lossTrain)
            MSE_test = self.MSE(lossTest)
            
            return W1, W2, lossTrain, lossTest, MSE_train, MSE_test

    #--------------------------------------------------------------------------------------------------------------#
    
    def computePartialDerivative(self, X, Y, W1, W2, lambda1, lambda2) :
    
        W1_W2 = np.dot(W2, W1)
        ffwd = (2/X.shape[0])*(np.dot(W1_W2, X).T - Y)        
        
        # for grad_1
        tmp_1 = np.dot(W2.T, ffwd.T)
        tmp_2 = np.dot(tmp_1, X.T)
        
        # for grad 2
        W1_XT = np.dot(W1, X) 
        
        # compute the final result
        grad_1 = tmp_2 + 2*lambda1*W1
        grad_2 = np.dot(ffwd.T, W1_XT.T) + 2*lambda2*W2
        
        
        return grad_1, grad_2

    #--------------------------------------------------------------------------------------------------------------#

    def computeLoss(self, X, Y, W1, W2, lambda1, lambda2) :
        
        W1_W2 = np.dot(W2, W1)
        ffwd = np.dot(W1_W2, X) - Y
        
        norm_1 =  (1/X.shape[0])*np.linalg.norm(ffwd) 
        norm_2 = lambda1*np.linalg.norm(W1) 
        norm_3 = lambda2*np.linalg.norm(W2)

        return norm_1 + norm_2 + norm_3

    #--------------------------------------------------------------------------------------------------------------#
    
    def MSE(self, loss):
        return np.sum((loss)**2)/(len(loss))
    
    #--------------------------------------------------------------------------------------------------------------#

    def getBestParams(self, lossTestingKFold, best_val):
   
        print ("len(lossTestingKFold) = ", len(lossTestingKFold))
        print ("len(best_val) = ", len(best_val))
        
        l1_, l2_, alpha, n_neuron = best_val[lossTestingKFold.index(min(lossTestingKFold))]
        print("Best lambda1 : ", l1_)
        print("Best lambda2 : ", l2_)
        print("Best alpha : ", alpha)
        print("Best n_neuron : ", n_neuron)
        
        print("The loss we had with those three params : ", lossTestingKFold[lossTestingKFold.index(min(lossTestingKFold))])
        
        return l1_, l2_, alpha, n_neuron
    
    #--------------------------------------------------------------------------------------------------------------#
