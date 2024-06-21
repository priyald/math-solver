import tensorflow as tf
import numpy as np

#fetch the data sample for training and testing
number_of_characters = 11
path = 'mnist.npz' 
with np.load(path, allow_pickle=True) as f:
    x_train, y_train = f['x_train'], f['y_train']
    x_test, y_test = f['x_test'], f['y_test']


#normalize the data
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)