#!/usr/bin/env python
# -*- coding:utf-8 -*-
__date__ = '2022.01.12'

import tensorflow as tf
from tensorflow.keras import datasets, layers, models, losses


VGG16 = [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"]
BATCH_SIZE = 128


def createModel(x_train, sgdLrOnly=False):
    model = models.Sequential()
    input_shape = x_train.shape[1:]
    first = True
    for config in VGG16:
        if isinstance(config, int):
            if first:
                model.add(layers.Conv2D(config, (3, 3), padding='same', input_shape=input_shape, use_bias=False))
                first = False
            else:
                model.add(layers.Conv2D(config, (3, 3), padding='same', use_bias=False))
            model.add(layers.BatchNormalization())
            model.add(layers.ReLU())
        elif config == "M":
            model.add(layers.MaxPooling2D((2, 2)))
        else:
            raise ValueError(f"Wrong config for VGG Net: {config}")

    model.add(layers.Flatten())
    model.add(layers.Dense(4096, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(4096, activation='relu'))
    model.add(layers.Dropout(0.5))
    # model.add(layers.Dense(10, activation='softmax'))
    if sgdLrOnly:
        model.add(layers.Dense(
            10, activation='softmax',
        ))
        opt = tf.keras.optimizers.SGD(learning_rate=5e-2)
    else:
        model.add(layers.Dense(
            10, activation='softmax',
            kernel_regularizer=tf.keras.regularizers.l2(1e-4),
            bias_regularizer=tf.keras.regularizers.l2(1e-4),
        ))
        opt = tf.keras.optimizers.SGD(learning_rate=1e-1, momentum=9e-1)
    return model, opt, losses.sparse_categorical_crossentropy


def loadData():
    (x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()
    x_train = x_train / 255.
    x_test = x_test / 255.
    print(x_train.dtype, y_train.dtype, x_test.dtype, y_test.dtype)
    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
    print(x_train.min(), x_train.max(), y_train.min(), y_train.max())
    return x_train, y_train, x_test, y_test, BATCH_SIZE

# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# conv2d (Conv2D)              (None, 32, 32, 64)        1728
# _________________________________________________________________
# batch_normalization (BatchNo (None, 32, 32, 64)        256
# _________________________________________________________________
# re_lu (ReLU)                 (None, 32, 32, 64)        0
# _________________________________________________________________
# conv2d_1 (Conv2D)            (None, 32, 32, 64)        36864
# _________________________________________________________________
# batch_normalization_1 (Batch (None, 32, 32, 64)        256
# _________________________________________________________________
# re_lu_1 (ReLU)               (None, 32, 32, 64)        0
# _________________________________________________________________
# max_pooling2d (MaxPooling2D) (None, 16, 16, 64)        0
# _________________________________________________________________
# conv2d_2 (Conv2D)            (None, 16, 16, 128)       73728
# _________________________________________________________________
# batch_normalization_2 (Batch (None, 16, 16, 128)       512
# _________________________________________________________________
# re_lu_2 (ReLU)               (None, 16, 16, 128)       0
# _________________________________________________________________
# conv2d_3 (Conv2D)            (None, 16, 16, 128)       147456
# _________________________________________________________________
# batch_normalization_3 (Batch (None, 16, 16, 128)       512
# _________________________________________________________________
# re_lu_3 (ReLU)               (None, 16, 16, 128)       0
# _________________________________________________________________
# max_pooling2d_1 (MaxPooling2 (None, 8, 8, 128)         0
# _________________________________________________________________
# conv2d_4 (Conv2D)            (None, 8, 8, 256)         294912
# _________________________________________________________________
# batch_normalization_4 (Batch (None, 8, 8, 256)         1024
# _________________________________________________________________
# re_lu_4 (ReLU)               (None, 8, 8, 256)         0
# _________________________________________________________________
# conv2d_5 (Conv2D)            (None, 8, 8, 256)         589824
# _________________________________________________________________
# batch_normalization_5 (Batch (None, 8, 8, 256)         1024
# _________________________________________________________________
# re_lu_5 (ReLU)               (None, 8, 8, 256)         0
# _________________________________________________________________
# conv2d_6 (Conv2D)            (None, 8, 8, 256)         589824
# _________________________________________________________________
# batch_normalization_6 (Batch (None, 8, 8, 256)         1024
# _________________________________________________________________
# re_lu_6 (ReLU)               (None, 8, 8, 256)         0
# _________________________________________________________________
# max_pooling2d_2 (MaxPooling2 (None, 4, 4, 256)         0
# _________________________________________________________________
# conv2d_7 (Conv2D)            (None, 4, 4, 512)         1179648
# _________________________________________________________________
# batch_normalization_7 (Batch (None, 4, 4, 512)         2048
# _________________________________________________________________
# re_lu_7 (ReLU)               (None, 4, 4, 512)         0
# _________________________________________________________________
# conv2d_8 (Conv2D)            (None, 4, 4, 512)         2359296
# _________________________________________________________________
# batch_normalization_8 (Batch (None, 4, 4, 512)         2048
# _________________________________________________________________
# re_lu_8 (ReLU)               (None, 4, 4, 512)         0
# _________________________________________________________________
# conv2d_9 (Conv2D)            (None, 4, 4, 512)         2359296
# _________________________________________________________________
# batch_normalization_9 (Batch (None, 4, 4, 512)         2048
# _________________________________________________________________
# re_lu_9 (ReLU)               (None, 4, 4, 512)         0
# _________________________________________________________________
# max_pooling2d_3 (MaxPooling2 (None, 2, 2, 512)         0
# _________________________________________________________________
# conv2d_10 (Conv2D)           (None, 2, 2, 512)         2359296
# _________________________________________________________________
# batch_normalization_10 (Batc (None, 2, 2, 512)         2048
# _________________________________________________________________
# re_lu_10 (ReLU)              (None, 2, 2, 512)         0
# _________________________________________________________________
# conv2d_11 (Conv2D)           (None, 2, 2, 512)         2359296
# _________________________________________________________________
# batch_normalization_11 (Batc (None, 2, 2, 512)         2048
# _________________________________________________________________
# re_lu_11 (ReLU)              (None, 2, 2, 512)         0
# _________________________________________________________________
# conv2d_12 (Conv2D)           (None, 2, 2, 512)         2359296
# _________________________________________________________________
# batch_normalization_12 (Batc (None, 2, 2, 512)         2048
# _________________________________________________________________
# re_lu_12 (ReLU)              (None, 2, 2, 512)         0
# _________________________________________________________________
# max_pooling2d_4 (MaxPooling2 (None, 1, 1, 512)         0
# _________________________________________________________________
# flatten (Flatten)            (None, 512)               0
# _________________________________________________________________
# dense (Dense)                (None, 4096)              2101248
# _________________________________________________________________
# dropout (Dropout)            (None, 4096)              0
# _________________________________________________________________
# dense_1 (Dense)              (None, 4096)              16781312
# _________________________________________________________________
# dropout_1 (Dropout)          (None, 4096)              0
# _________________________________________________________________
# dense_2 (Dense)              (None, 10)                40970
# =================================================================
# Total params: 33,650,890
# Trainable params: 33,642,442
# Non-trainable params: 8,448
