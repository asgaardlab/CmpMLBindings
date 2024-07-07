#!/usr/bin/env python
# -*- coding:utf-8 -*-
__date__ = '2021.10.29'


import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets, layers, models, losses


BATCH_SIZE = 128


def createLeNet5Model(x_train, sgdLrOnly=False):
    model = models.Sequential()
    model.add(layers.Conv2D(6, 5, activation='tanh', input_shape=x_train.shape[1:]))
    model.add(layers.MaxPool2D(2))
    model.add(layers.Conv2D(16, 5, activation='tanh'))
    model.add(layers.MaxPool2D(2))
    model.add(layers.Conv2D(120, 5, activation='tanh'))
    model.add(layers.Flatten())
    model.add(layers.Dense(84, activation='tanh'))
    model.add(layers.Dense(10, activation="softmax"))

    if sgdLrOnly:
        opt = tf.keras.optimizers.SGD(learning_rate=5e-2)
    else:
        opt = tf.keras.optimizers.SGD(learning_rate=5e-2, momentum=9e-1)
    return model, opt, losses.sparse_categorical_crossentropy

# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# conv2d (Conv2D)              (None, 28, 28, 6)         156
# _________________________________________________________________
# max_pooling2d (MaxPooling2D) (None, 14, 14, 6)         0
# _________________________________________________________________
# conv2d_1 (Conv2D)            (None, 10, 10, 16)        2416
# _________________________________________________________________
# max_pooling2d_1 (MaxPooling2 (None, 5, 5, 16)          0
# _________________________________________________________________
# conv2d_2 (Conv2D)            (None, 1, 1, 120)         48120
# _________________________________________________________________
# flatten (Flatten)            (None, 120)               0
# _________________________________________________________________
# dense (Dense)                (None, 84)                10164
# _________________________________________________________________
# dense_1 (Dense)              (None, 10)                850
# =================================================================
# Total params: 61,706
# Trainable params: 61,706
# Non-trainable params: 0


def createLeNet1Model(x_train, sgdLrOnly=False):
    model = models.Sequential()
    model.add(layers.Conv2D(4, 5, activation='tanh', input_shape=x_train.shape[1:]))
    model.add(layers.MaxPool2D(2))
    model.add(layers.Conv2D(12, 5, activation='tanh'))
    model.add(layers.MaxPool2D(2))
    model.add(layers.Flatten())
    model.add(layers.Dense(10, activation="softmax"))

    if sgdLrOnly:
        opt = tf.keras.optimizers.SGD(learning_rate=5e-2)
    else:
        opt = tf.keras.optimizers.SGD(learning_rate=5e-2, momentum=9e-1)
    return model, opt, losses.sparse_categorical_crossentropy

# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# conv2d (Conv2D)              (None, 28, 28, 4)         104
# _________________________________________________________________
# max_pooling2d (MaxPooling2D) (None, 14, 14, 4)         0
# _________________________________________________________________
# conv2d_1 (Conv2D)            (None, 10, 10, 12)        1212
# _________________________________________________________________
# max_pooling2d_1 (MaxPooling2 (None, 5, 5, 12)          0
# _________________________________________________________________
# flatten (Flatten)            (None, 300)               0
# _________________________________________________________________
# dense (Dense)                (None, 10)                3010
# =================================================================
# Total params: 4,326
# Trainable params: 4,326
# Non-trainable params: 0



def loadData():
    (x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
    # train_set = np.hstack((y_train.reshape(60000, -1), x_train.reshape(60000, -1)))
    # test_set = np.hstack((y_test.reshape(y_test.shape[0], -1), x_test.reshape(x_test.shape[0], -1)))
    # print(train_set.shape, test_set.shape)
    # print(np.allclose(train_set[:, 0], y_train))
    # print(np.allclose(test_set[:, 0], y_test))
    x_train = tf.pad(x_train, [[0, 0], [2, 2], [2, 2]]) / 255
    x_test = tf.pad(x_test, [[0, 0], [2, 2], [2, 2]]) / 255
    x_train = tf.expand_dims(x_train, axis=3, name=None)
    x_test = tf.expand_dims(x_test, axis=3, name=None)
    print(x_train.dtype, y_train.dtype, x_test.dtype, y_test.dtype)
    return x_train, y_train, x_test, y_test, BATCH_SIZE
