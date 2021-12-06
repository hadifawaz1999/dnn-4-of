import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from tqdm import tqdm

class SGD(Optimizer) :

    #--------------------------------------------------------------------------------------------------------------#

    def __init__(self):
        super()

    #--------------------------------------------------------------------------------------------------------------#

    def create_mini_batches(self, X, y, batch_size):
    
        mini_batches = [] 
        data = np.hstack((X, y)) 
        np.random.shuffle(data) 
        n_minibatches = data.shape[0] // batch_size 
        
        i = 0
    
        for i in range(n_minibatches + 1): 
            mini_batch = data[i * batch_size:(i + 1)*batch_size, :] 
            X_mini = mini_batch[:, :-1] 
            Y_mini = mini_batch[:, -1].reshape((-1, 1)) 
            mini_batches.append((X_mini, Y_mini))
            
        if data.shape[0] % batch_size != 0: 
            mini_batch = data[i * batch_size:data.shape[0]] 
            X_mini = mini_batch[:, :-1] 
            Y_mini = mini_batch[:, -1].reshape((-1, 1)) 
            mini_batches.append((X_mini, Y_mini))
            
        return mini_batches 

    #----------------------------------------------------------------------------------------------------------------#

    def miniBatchSGD(self, X_train, X_test, y_train, y_test, W1, W2, lambda1, lambda2, alpha, N, batch_size, with_print) :
    
        lossTrain = np.zeros(N)
        lossTest = np.zeros(N)
        
        # just to avoid prints when we do cross validation
        if(with_print) :
            
            for i in tqdm(range(N)) :
                    
                mini_batches_train = create_mini_batches(X_train, y_train, batch_size)

                for mini_batch in mini_batches_train:

                    X_train, y_train = mini_batch 
                    
                    lossTrain_ = self.computeLoss(X_train.T, y_train, W1, W2, lambda1, lambda2)
                    lossTest_ = self.computeLoss(X_test.T, y_test, W1, W2, lambda1, lambda2)

                    lossTrain[i] = lossTrain_
                    lossTest[i] = lossTest_
                    
                    grad_1, grad_2 = self.computePartialDerivative(X_train.T, y_train, W1, W2, lambda1, lambda2)

                    W1 = W1 - alpha*grad_1
                    W2 = W2 - alpha*grad_2
                    
            MSE_train = self.MSE(lossTrain)
            MSE_test = self.MSE(lossTest)

            print("W2.shape : ", W2.shape)
            print("W1.shape : ", W1.shape)
            print("X.shape : ", X.shape)
            print("Test-Loss mean : ", lossTest.mean())
            print("Last Test loss value : ", lossTest[-1])
            print("MSE_test : ", MSE_test)
            
            return W1, W2, lossTrain, lossTest, MSE_train, MSE_test
        
        else :
            
            for i in range(N) :
                    
                mini_batches_train = create_mini_batches(X_train, y_train, batch_size)

                for mini_batch in mini_batches_train:

                    X_train, y_train = mini_batch 
                    
                    lossTrain_ = self.computeLoss(X_train.T, y_train, W1, W2, lambda1, lambda2)
                    lossTest_ = self.computeLoss(X_test.T, y_test, W1, W2, lambda1, lambda2)

                    lossTrain[i] = lossTrain_
                    lossTest[i] = lossTest_

                    grad_1, grad_2 = self.computePartialDerivative(X_train.T, y_train, W1, W2, lambda1, lambda2)

                    W1 = W1 - alpha*grad_1
                    W2 = W2 - alpha*grad_2
            
            MSE_train = self.MSE(lossTrain)
            MSE_test = self.MSE(lossTest)
            
            return W1, W2, lossTrain, lossTest, MSE_train, MSE_test
    #----------------------------------------------------------------------------------------------------------------#

    def KFoldCrossValidation_SGD(self, df, k, X, Y, lambda1, lambda2, N, min_neurons, max_neurons,alphas, batch_size, with_print=0):
    
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
                            W1, W2, lossTrain, lossTest, MSE_train, MSE_test = miniBatchSGD(X_train, X_test, y_train, y_test, W1, W2, l1, l2, alpha, N, batch_size, with_print)

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

    #----------------------------------------------------------------------------------------------------------------#