import tensorflow as tf
import numpy as np
from numData import x_train, y_train



#choosing sequential model
model = tf.keras.models.Sequential()


#let's create the layers for our model
model.add(tf.keras.layers.Input(shape=(28,28)))
#first layer flattens the grid from 28x28 to a long 1x789 way
model.add(tf.keras.layers.Flatten())
#dense layer represents a basic neural network layer that is connected completely to all previous layers input
#ReLU function is 𝑓(𝑥)=max(0,𝑥)
model.add(tf.keras.layers.Dense(128, activation="relu"))
model.add(tf.keras.layers.Dense(128, activation="relu"))
model.add(tf.keras.layers.Dense(128, activation="relu"))
#final output layer
#it represents the 10 different digit. hence it has 10 different values as output
#softmax functions makes sure that all the neuron values add up to 1
model.add(tf.keras.layers.Dense(10, activation="softmax"))


#compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=10)
model.save('numMod.keras')

