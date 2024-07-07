#!/usr/bin/env python
# -*- coding:utf-8 -*-

import pathlib
import pandas as pd
import numpy as np
from tensorflow.keras import datasets
import tensorflow as tf
import tensorflowjs as tfjs
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence


OUT_PATH = pathlib.Path("./out/tensorflow")
TF_LENET_PY_OUT_PATH = pathlib.Path("./out/tensorflow/lenet5/py")

MNIST_NPY_PATH = pathlib.Path("./data/mnist-npy")
MNIST_NPY_PATH.mkdir(parents=True, exist_ok=True)

CIFAR_CSV_PATH = pathlib.Path("./data/cifar-10-csv")
CIFAR_CSV_PATH.mkdir(parents=True, exist_ok=True)
CIFAR_TXT_PATH = pathlib.Path("./data/cifar-10-txt")
CIFAR_TXT_PATH.mkdir(parents=True, exist_ok=True)
CIFAR_NPY_PATH = pathlib.Path("./data/cifar-10-npy")
CIFAR_NPY_PATH.mkdir(parents=True, exist_ok=True)

TFJS_OUT = OUT_PATH / "tfjs"
TFJS_OUT.mkdir(parents=True, exist_ok=True)
TF_FMT_OUT = OUT_PATH / "tf_fmt"
TF_FMT_OUT.mkdir(parents=True, exist_ok=True)
IMDB_DATA_PATH = pathlib.Path("./data/imdb").absolute()
IMDB_DATA_PATH.mkdir(parents=True, exist_ok=True)


def transMNISTFormat():
    (x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()

    train_set = np.hstack((y_train.reshape(60000, -1), x_train.reshape(60000, -1)))
    test_set = np.hstack((y_test.reshape(y_test.shape[0], -1), x_test.reshape(x_test.shape[0], -1)))
    print(train_set.shape, test_set.shape)
    print(np.allclose(train_set[:, 0], y_train))
    print(np.allclose(test_set[:, 0], y_test))

    train_set = pd.DataFrame(train_set, columns=["label"] + [f"pixel{i}" for i in range(28 * 28)])
    train_set.to_csv("./data/MNIST/train.csv", index=False)
    test_set = pd.DataFrame(test_set, columns=["label"] + [f"pixel{i}" for i in range(28 * 28)])
    test_set.to_csv("./data/MNIST/test.csv", index=False)

    x_train = tf.pad(x_train, [[0, 0], [2, 2], [2, 2]])
    x_test = tf.pad(x_test, [[0, 0], [2, 2], [2, 2]])
    x_train_saved = x_train.numpy().reshape(x_train.shape[0], -1)
    x_test_saved = x_test.numpy().reshape(x_test.shape[0], -1)
    print(x_train_saved.shape)
    print(y_train.shape)
    print(x_test_saved.shape)
    print(y_test.shape)
    np.savetxt("./data/MNIST/x_train_padded.txt", x_train_saved, fmt="%d")
    np.savetxt("./data/MNIST/y_train_padded.txt", y_train, fmt="%d")
    np.savetxt("./data/MNIST/x_test_padded.txt", x_test_saved, fmt="%d")
    np.savetxt("./data/MNIST/y_test_padded.txt", y_test, fmt="%d")

    np.save(MNIST_NPY_PATH / "x_train_padded.npy", x_train_saved.astype(np.int32))
    np.save(MNIST_NPY_PATH / "x_test_padded.npy", x_test_saved.astype(np.int32))
    np.save(MNIST_NPY_PATH / "y_test.npy", y_test.astype(np.int32))
    np.save(MNIST_NPY_PATH / "y_train.npy", y_train.astype(np.int32))


def transCIFARFormat():
    def saveCSV(x_data, split_to=5, name="x_train"):
        x_data_flat = x_data.reshape(x_data.shape[0] * 3, -1)
        print(x_data_flat.shape)
        slice_size = x_data_flat.shape[0] // split_to
        print(f"split to {split_to}, slice_size: {slice_size}")
        for i in range(split_to):
            x_data_split = pd.DataFrame(
                x_data_flat[slice_size * i: slice_size * (i + 1), :],
                columns=[f"pixel{i}" for i in range(32 * 32)]
            )
            x_data_split.to_csv(CIFAR_CSV_PATH / f"{name}_{i}.csv", index=False)

    split_to = 5
    (x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()
    print(x_train.shape, y_train.shape)

    # ========== for PyTorch csv ========== #
    x_train_channel_first = np.rollaxis(x_train, 3, 1)
    x_test_channel_first = np.rollaxis(x_test, 3, 1)
    print("channel first shape:", x_train_channel_first.shape, x_test_channel_first.shape)
    saveCSV(x_train_channel_first, split_to, "x_train_channel_first")
    x_test_channel_first_df = pd.DataFrame(x_test_channel_first.reshape(x_test_channel_first.shape[0] * 3, -1), columns=[f"pixel{i}" for i in range(32 * 32)])
    x_test_channel_first_df.to_csv(CIFAR_CSV_PATH / "x_test_channel_first.csv", index=False)
    np.save(CIFAR_NPY_PATH / "x_train_channel_first.npy", x_train_channel_first.reshape(x_train_channel_first.shape[0], -1))
    np.save(CIFAR_NPY_PATH / "x_test_channel_first.npy", x_test_channel_first.reshape(x_test_channel_first.shape[0], -1))

    # ========== for TensorFlow txt ========== #
    x_train_saved = x_train.reshape(x_train.shape[0], -1)
    x_test_saved = x_test.reshape(x_test.shape[0], -1)
    print(x_train_saved.shape)
    print(y_train.shape)
    print(x_test_saved.shape)
    print(y_test.shape)

    assert x_train_saved.shape[0] % split_to == 0
    slice_size = x_train_saved.shape[0] // split_to
    print(f"split to {split_to}, slice_size: {slice_size}")
    for i in range(split_to):
        np.savetxt(CIFAR_TXT_PATH / f"x_train_{i}.txt", x_train_saved[i * slice_size : (i + 1) * slice_size], fmt="%d")
    np.savetxt(CIFAR_TXT_PATH / "x_train_all.txt", x_train_saved, fmt="%d")
    np.savetxt(CIFAR_TXT_PATH / "y_train.txt", y_train, fmt="%d")
    np.savetxt(CIFAR_TXT_PATH / "x_test.txt", x_test_saved, fmt="%d")
    np.savetxt(CIFAR_TXT_PATH / "y_test.txt", y_test, fmt="%d")
    np.save(CIFAR_NPY_PATH / "x_train.npy", x_train_saved.astype(np.int32))
    np.save(CIFAR_NPY_PATH / "x_test.npy", x_test_saved.astype(np.int32))
    np.save(CIFAR_NPY_PATH / "y_train.npy", y_train.astype(np.int32))
    np.save(CIFAR_NPY_PATH / "y_test.npy", y_test.astype(np.int32))

    # ========== for TensorFlow csv ========== #
    assert x_train.shape[0] % split_to == 0
    saveCSV(x_train, split_to=split_to)

    y_train = pd.DataFrame(y_train.reshape(y_train.shape[0], -1), columns=["label", ])
    y_train.to_csv(CIFAR_CSV_PATH / "y_train.csv", index=False)

    x_test = pd.DataFrame(x_test.reshape(x_test.shape[0] * 3, -1), columns=[f"pixel{i}" for i in range(32 * 32)])
    x_test.to_csv(CIFAR_CSV_PATH / "x_test.csv", index=False)
    y_test = pd.DataFrame(y_test.reshape(y_test.shape[0], -1), columns=["label", ])
    y_test.to_csv(CIFAR_CSV_PATH / "y_test.csv", index=False)


def transIMDbFormat():
    num_words = 10000
    (x_train, y_train), (x_test, y_test) = imdb.load_data(IMDB_DATA_PATH / "imdb.npz", num_words=num_words, seed=113)
    max_review_length = 300
    x_train = sequence.pad_sequences(x_train, maxlen=max_review_length)
    x_test = sequence.pad_sequences(x_test, maxlen=max_review_length)
    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
    np.savetxt(IMDB_DATA_PATH / "x_train.txt", x_train, fmt="%d")
    np.savetxt(IMDB_DATA_PATH / "y_train.txt", y_train, fmt="%d")
    np.savetxt(IMDB_DATA_PATH / "x_test.txt", x_test, fmt="%d")
    np.savetxt(IMDB_DATA_PATH / "y_test.txt", y_test, fmt="%d")

    train_set = np.hstack((y_train.reshape(y_train.shape[0], -1), x_train.reshape(y_train.shape[0], -1)))
    test_set = np.hstack((y_test.reshape(y_test.shape[0], -1), x_test.reshape(x_test.shape[0], -1)))
    print(train_set.shape, test_set.shape)
    print(np.allclose(train_set[:, 0], y_train))
    print(np.allclose(test_set[:, 0], y_test))

    train_set = pd.DataFrame(train_set, columns=["label"] + [f"w{i}" for i in range(max_review_length)])
    train_set.to_csv(IMDB_DATA_PATH / "train.csv", index=False)
    test_set = pd.DataFrame(test_set, columns=["label"] + [f"w{i}" for i in range(max_review_length)])
    test_set.to_csv(IMDB_DATA_PATH / "test.csv", index=False)


def trainsModelToJS(model_name):
    model_path = OUT_PATH / f"{model_name}/py"
    for i in range(5):
        model = tf.keras.models.load_model(model_path / f"mnist-{model_name}_{i}")
        tfjs.converters.save_keras_model(model, model_path / f"mnist-{model_name}_{i}_tfjs")


def readH5():
    import h5py
    f = h5py.File("./out/tensorflow/lenet5/mnist_weights_0.h5", "r")
    # f = h5py.File("./out/tensorflow/lenet5/test_weights.h5", "r")
    print(f"Keras version: {f.attrs['keras_version']}")
    for key in f.keys():
        print(key)  # Names of the groups in HDF5 file.
        # Get the HDF5 group
        group = f[key]
        # Checkout what keys are inside that group.
        for key in group.keys():
            print("\t", key, group[key])
            for key2 in group[key].keys():
                print("\t\t", key, key2, group[key][key2])
            # Do whatever you want with data
            # print(data)

    # After you are done
    f.close()


def extractDiff():
    data_path = pathlib.Path("/home/leo/Documents/LearningInUA/Papers/2021/experiment/wip-21-lihao-compare_ml_bindings-code/tensorflow_bindings/rs/lenet_deploy_rs/")
    res_a = np.loadtxt(data_path / "res_1.txt").tolist()
    res_b = np.loadtxt(data_path / "res_100.txt").tolist()
    diff_idx = []
    for idx, (a, b) in enumerate(zip(res_a, res_b)):
        if a != b:
            diff_idx.append(idx)
    print(diff_idx)
    source_data_path = pathlib.Path("/home/leo/Documents/LearningInUA/Papers/2021/experiment/wip-21-lihao-compare_ml_bindings-code/data/MNIST")
    saved_to_path = pathlib.Path("/home/leo/Documents/LearningInUA/Papers/2021/experiment/wip-21-lihao-compare_ml_bindings-code/tensorflow_bindings/rs/lenet_deploy_rs/")
    data_files_for_extract = [source_data_path / "x_test_padded.txt", source_data_path / "y_test_padded.txt"]
    for p in data_files_for_extract:
        data = np.loadtxt(p)[diff_idx]
        np.savetxt(saved_to_path / "{}_diff{}".format(p.stem, p.suffix), data, fmt="%d")


def extractSpecificData():
    source_data_path = pathlib.Path("/home/leo/Documents/LearningInUA/Papers/2021/experiment/wip-21-lihao-compare_ml_bindings-code/data/MNIST")
    saved_to_path = source_data_path
    # saved_to_path = pathlib.Path("/home/leo/Documents/LearningInUA/Papers/2021/experiment/wip-21-lihao-compare_ml_bindings-code/tensorflow_bindings/rs/lenet_deploy_rs/")
    data_files_for_extract = [source_data_path / "x_test_padded.txt", source_data_path / "y_test_padded.txt"]
    diff_idx = list(range(0 + 100*17, 100 * 18))
    for p in data_files_for_extract:
        data = np.loadtxt(p)[diff_idx]
        np.savetxt(saved_to_path / "{}_diff{}".format(p.stem, p.suffix), data, fmt="%d")


def main():
    transCIFARFormat()
    transMNISTFormat()
    # transIMDbFormat()
    # trainsModelToJS()
    # readH5()
    # extractDiff()
    # extractSpecificData()


if __name__ == '__main__':
    main()
