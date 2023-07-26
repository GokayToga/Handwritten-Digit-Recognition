import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

#data seperation
mnsit = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnsit.load_data()

x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

#creating the model and training it and saving it
#model = tf.keras.models.Sequential()
#model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
#model.add(tf.keras.layers.Dense(units=128, activation=tf.nn.relu))
#model.add(tf.keras.layers.Dense(units=128, activation=tf.nn.relu))
#model.add(tf.keras.layers.Dense(units=10, activation=tf.nn.softmax))

#model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

#model.fit(x_train, y_train, epochs=3)

#model.save('digits.model')

model = tf.keras.models.load_model('digits.model')

#seeing accuracy and loss for the model
#loss, accuracy = model.evaluate(x_test, y_test)

#print(accuracy)
#print(loss)