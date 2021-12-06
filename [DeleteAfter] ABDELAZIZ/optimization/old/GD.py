import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from tqdm import tqdm

class GD(Optimizer) :

    #--------------------------------------------------------------------------------------------------------------#

    def __init__(self):
        super()

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

    def KFoldCrossValidation_BP(self, df, k, X, Y, lambda1, lambda2, N, min_neurons, max_neurons,alphas, with_print=0):
    
        ts = int(len(X)/k)
        loss = []
        mse = []
        best_val = []
        neurons = range(min_neurons,max_neurons+1)
        
        #df = getData()
        
        
        
        for i in tqdm(range(k)) :
            # fix number of neurons
            for n_neuron in neurons:
                # fix learning rate
                for alpha in alphas :
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
                            W1, W2, lossTrain, lossTest, MSE_train, MSE_test = backPropagation(X_train.T, X_test.T, y_train, y_test, W1, W2, l1, l2, alpha, N, with_print)
                            
                            # cost
                            real_loss = self.computeLoss(X_test.T, y_test, W1, W2, l1, l2)  
                            loss.append(real_loss)
                            
                            loss_arr = np.array(loss)
                            real_mse = self.MSE(loss_arr)
                            mse.append(real_mse)

                            # hyper params
                            best_val.append((l1,l2,alpha, n_neuron))
                    

        # return the best two params l1, l2
        l1_, l2_, alpha_, n_neuron_ = best_val[loss.index(min(loss))]
        return loss, mse, best_val, l1_, l2_, alpha_, n_neuron_
        
    #--------------------------------------------------------------------------------------------------------------#
    