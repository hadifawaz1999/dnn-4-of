import tensorflow as tf
import numpy as np

tf.config.run_functions_eagerly(True)


def constellation_loss_function(alpha):

    def temp_function(ytrue, ypred):

        ytrue = tf.keras.backend.cast(ytrue, dtype=ytrue.dtype)

        ypred = tf.keras.backend.cast(ypred, dtype=ypred.dtype)

        # distance_to_centers = alpha * tf.math.abs(tf.math.subtract(ypred, ytrue))
        
        distance_to_centers = tf.math.abs(tf.math.subtract(ypred,ytrue))

        # distance_to_centers = tf.keras.activations.relu(distance_to_centers,threshold=(0.1-1e-9)*alpha)

        relu_1 = tf.keras.activations.relu(distance_to_centers)
        
        relu_2 = tf.keras.activations.relu(distance_to_centers,threshold=0.1)
        
        relu_3 = tf.keras.activations.relu(distance_to_centers,threshold=0.1)
        
        distance_to_centers = relu_1 - relu_2 + alpha * relu_3

        # loss = tf.math.reduce_mean(tf.square(distance_to_centers))
        
        loss = tf.math.reduce_mean(distance_to_centers)

        return loss

    return temp_function