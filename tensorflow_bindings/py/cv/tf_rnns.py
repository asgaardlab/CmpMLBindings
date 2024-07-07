#!/usr/bin/env python
# -*- coding:utf-8 -*-
__date__ = '2022.01.12'

import pathlib
import numpy as np

import tensorflow as tf
from tensorflow.keras import layers, models, losses

IMDB_DATA_PATH = pathlib.Path("../../../data/imdb")
NUM_WORDS = 10000
EMBEDDING_VEC_LEN = 300
BATCH_SIZE = 256


def createLSTMModel(x_train, _):
    sentence_len = x_train.shape[1]
    model = models.Sequential()
    model.add(layers.Embedding(input_dim=NUM_WORDS, output_dim=EMBEDDING_VEC_LEN, input_length=sentence_len))
    # model.add(layers.Dropout(0.5))
    model.add(layers.LSTM(512))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(units=1, activation='sigmoid'))

    opt = tf.keras.optimizers.Adam(learning_rate=8e-5)
    return model, opt, losses.binary_crossentropy

# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# embedding (Embedding)        (None, 300, 300)          3000000
# _________________________________________________________________
# lstm (LSTM)                  (None, 512)               1665024
# _________________________________________________________________
# dropout (Dropout)            (None, 512)               0
# _________________________________________________________________
# dense (Dense)                (None, 1)                 513
# =================================================================
# Total params: 4,665,537
# Trainable params: 4,665,537
# Non-trainable params: 0


def createGRUModel(x_train, _):
    sentence_len = x_train.shape[1]
    model = models.Sequential()
    model.add(layers.Embedding(input_dim=NUM_WORDS, output_dim=EMBEDDING_VEC_LEN, input_length=sentence_len))
    # model.add(layers.Dropout(0.5))
    model.add(layers.GRU(512))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(units=1, activation='sigmoid'))

    opt = tf.keras.optimizers.Adam(learning_rate=3e-4)
    return model, opt, losses.binary_crossentropy

# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# embedding (Embedding)        (None, 300, 300)          3000000
# _________________________________________________________________
# gru (GRU)                    (None, 512)               1250304
# _________________________________________________________________
# dropout (Dropout)            (None, 512)               0
# _________________________________________________________________
# dense (Dense)                (None, 1)                 513
# =================================================================
# Total params: 4,250,817
# Trainable params: 4,250,817
# Non-trainable params: 0


def createGRUResetBeforeModel(x_train, _):
    sentence_len = x_train.shape[1]
    model = models.Sequential()
    model.add(layers.Embedding(input_dim=NUM_WORDS, output_dim=EMBEDDING_VEC_LEN, input_length=sentence_len))
    # model.add(layers.Dropout(0.5))
    model.add(layers.GRU(512, reset_after=False))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(units=1, activation='sigmoid'))

    opt = tf.keras.optimizers.Adam(learning_rate=3e-4)
    return model, opt, losses.binary_crossentropy


def createTextCNNModel(x_train, _):
    model = TextCNN(x_train)
    model.build(x_train.shape)
    opt = tf.keras.optimizers.Adam(learning_rate=8e-5)
    return model, opt, losses.binary_crossentropy


class TextCNN(tf.keras.Model):
    FILTER_SIZES = [2, 3, 4, 5]
    NUM_FILTERS = 256

    def __init__(self, x_input):
        super(TextCNN, self).__init__()

        self.emb = layers.Embedding(input_dim=NUM_WORDS, output_dim=EMBEDDING_VEC_LEN, input_length=x_input.shape[1])
        self.convs = []
        self.max_pools = []
        for fs in self.__class__.FILTER_SIZES:
            self.convs.append(layers.Conv2D(self.__class__.NUM_FILTERS, (fs, EMBEDDING_VEC_LEN), activation="relu"))
            self.max_pools.append(layers.MaxPool2D((x_input.shape[1] - fs + 1, 1), (1, 1)))

        self.dropout = layers.Dropout(0.5)
        self.flatten = layers.Flatten()
        self.dense = layers.Dense(1, activation='sigmoid')

    def call(self, inputs):
        x_emb = self.emb(inputs)
        x_emb_reshaped = tf.expand_dims(x_emb, -1)
        # print("x_emb_reshaped:", x_emb_reshaped.shape)

        layers_out = []
        for conv, max_p in zip(self.convs, self.max_pools):
            t = conv(x_emb_reshaped)
            # print("t:", t.shape)
            o = max_p(t)
            layers_out.append(o)
            # print("o:", o.shape)
        concat_out = tf.concat(layers_out, 3)
        # print("concat_out:", concat_out.shape)
        concat_out = self.dropout(concat_out)
        concat_out_flat = self.flatten(concat_out)
        # print("concat_out_flat:", concat_out_flat.shape)
        out = self.dense(concat_out_flat)
        return out


###
# x_emb_reshaped: (25000, 300, 300, 1)
# t: (25000, 299, 1, 256)
# o: (25000, 1, 1, 256)
# t: (25000, 298, 1, 256)
# o: (25000, 1, 1, 256)
# t: (25000, 297, 1, 256)
# o: (25000, 1, 1, 256)
# t: (25000, 296, 1, 256)
# o: (25000, 1, 1, 256)
# concat_out: (25000, 1, 1, 1024)
# concat_out_flat: (25000, 1024)
# Model: "text_cnn"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# embedding (Embedding)        multiple                  3000000
# _________________________________________________________________
# conv2d (Conv2D)              multiple                  153856
# _________________________________________________________________
# conv2d_1 (Conv2D)            multiple                  230656
# _________________________________________________________________
# conv2d_2 (Conv2D)            multiple                  307456
# _________________________________________________________________
# conv2d_3 (Conv2D)            multiple                  384256
# _________________________________________________________________
# max_pooling2d (MaxPooling2D) multiple                  0
# _________________________________________________________________
# max_pooling2d_1 (MaxPooling2 multiple                  0
# _________________________________________________________________
# max_pooling2d_2 (MaxPooling2 multiple                  0
# _________________________________________________________________
# max_pooling2d_3 (MaxPooling2 multiple                  0
# _________________________________________________________________
# dropout (Dropout)            multiple                  0
# _________________________________________________________________
# flatten (Flatten)            multiple                  0
# _________________________________________________________________
# dense (Dense)                multiple                  1025
# =================================================================
# Total params: 4,077,249
# Trainable params: 4,077,249
# Non-trainable params: 0
###


def loadData():
    x_train = np.loadtxt(IMDB_DATA_PATH / "x_train.txt", dtype=np.int32)
    y_train = np.loadtxt(IMDB_DATA_PATH / "y_train.txt", dtype=np.int64)
    x_test = np.loadtxt(IMDB_DATA_PATH / "x_test.txt", dtype=np.int32)
    y_test = np.loadtxt(IMDB_DATA_PATH / "y_test.txt", dtype=np.int64)
    print(x_train.dtype, y_train.dtype, x_test.dtype, y_test.dtype)
    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
    return x_train, y_train, x_test, y_test, BATCH_SIZE
