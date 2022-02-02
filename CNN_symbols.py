import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from custom_loss_function import constellation_loss_function

gpu_path = ''

class CNN_symbols:

    def __init__(self, xtrain, ytrain, xtest, ytest, xvalidation, yvalidation,
                 batch_size=200, epochs=300, learning_rate=0.05,
                 build_model=True, save_model=True, draw_model=True,
                 show_summary=True, show_verbose=True):

        self.xtrain = xtrain
        self.ytrain = ytrain
        self.xtest = xtest
        self.ytest = ytest
        self.xvalidation = xvalidation
        self.yvalidation = yvalidation

        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.build_model = build_model
        self.save_model = save_model
        self.draw_model = draw_model
        self.show_summary = show_summary
        self.show_verbose = show_verbose

        if self.build_model:

            self.build_my_model()


    def build_my_model(self):

        self.N = int(self.xtrain.shape[1])
 
        self.input_layer = tf.keras.layers.Input(
            self.xtrain.shape[1:], name='Input')

        self.residual = self.input_layer

        self.conv1 = tf.keras.layers.Conv1D(
            filters=2, kernel_size=32 ,activation='relu',padding="same", name='Conv1')(self.input_layer)

        self.conv2 = tf.keras.layers.Conv1D(
           filters=2,kernel_size=32, activation='relu',padding="same", name='Conv2')(self.conv1)

        

        #self.residual = tf.keras.layers.Conv1D(filters=2,kernel_size=1,activation='linear')(self.residual)
        
        

        #self.results = tf.keras.layers.add([self.BN,self.residual])
        
        self.flatten1 = tf.keras.layers.Flatten(name='Flatten1')(self.conv2)
        
        self.flatten2 = tf.keras.layers.Flatten(name='Flatten2')(self.residual)
        
        

        self.results = tf.keras.layers.add([self.flatten1,self.flatten2])
        
        self.output_layer = tf.keras.layers.Dense(
           units=64, activation='linear', name='Output')(self.results)

        self.my_model = tf.keras.models.Model(
            inputs=self.input_layer, outputs=self.output_layer)

        # self.my_loss = tf.keras.losses.MeanAbsoluteError()
        
        self.my_loss = constellation_loss_function(alpha=1.1)

        self.my_optimizer = tf.keras.optimizers.SGD(lr=self.learning_rate)

        self.my_model.compile(loss=self.my_loss, optimizer=self.my_optimizer)


        if self.show_summary:

            self.my_model.summary()
        
        if self.draw_model:
            
            tf.keras.utils.plot_model(self.my_model,gpu_path+'Predictors/CNN_symbols/CNN_symbols.png',show_shapes=True)

    def fit(self):

        self.history = self.my_model.fit(x=self.xtrain, y=self.ytrain,
                                         batch_size=self.batch_size, epochs=self.epochs,
                                         verbose=self.show_verbose, validation_data=(self.xvalidation,
                                                                                     self.yvalidation))
        
        self.loss = self.history.history['loss']
        
        self.val_loss = self.history.history['val_loss']

        plt.figure(figsize=(20,10))
        
        plt.plot(self.loss,color='blue',lw=3,label="train loss")
        plt.plot(self.val_loss,color='red',lw=3,label='val loss')
        
        plt.legend()
        
        plt.savefig(gpu_path+'Predictors/CNN_symbols/CNN_symbols_loss_val_train.png')
        
        plt.clf()
        
    def evaluation(self):
        
        self.ypred = self.my_model.predict(self.xvalidation)
        
        self.ypred_train = self.my_model.predict(self.xtrain)
        
        self.error = np.mean((self.yvalidation - self.ypred)**2)
        
        self.error_train = np.mean((self.ytrain - self.ypred_train)**2)
        
        return self.error
        
        # return self.error_train