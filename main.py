import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

mnsit = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnsit.load_data()

x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)