#!/usr/bin/env python
# -*- coding:utf-8 -*-

import argparse
from enum import Enum

import os
import time
import pathlib
import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets, layers, models, callbacks
import json
import random

import tf_lenet
import tf_vgg
import tf_resnet
import tf_rnns


OUT_PATH = pathlib.Path("../../../out/tensorflow")
OUT_LR_ONLY_PATH = pathlib.Path("../../../out/tensorflow_lr_only")
SEEDS = np.loadtxt("../../../random_seeds.txt").astype(np.uint32)


class DatasetName(Enum):
    CIFAR = "cifar"
    MNIST = "mnist"
    IMDb = "imdb"


class Model(Enum):
    VGG16 = "vgg16"
    ResNet20 = "resnet20"
    LeNet1 = "lenet1"
    LeNet5 = "lenet5"
    LSTM = "lstm"
    GRU = "gru"
    GRURB = "grurb"
    TextCNN = "textcnn"


Model2DatasetName = {
    Model.VGG16: DatasetName.CIFAR,
    Model.ResNet20: DatasetName.CIFAR,
    Model.LeNet5: DatasetName.MNIST,
    Model.LeNet1: DatasetName.MNIST,
    Model.LSTM: DatasetName.IMDb,
    Model.GRU: DatasetName.IMDb,
    Model.GRURB: DatasetName.IMDb,
    Model.TextCNN: DatasetName.IMDb,
}


ModelMap = {
    Model.VGG16: tf_vgg.createModel,
    Model.ResNet20: tf_resnet.createModel,
    Model.LeNet5: tf_lenet.createLeNet5Model,
    Model.LeNet1: tf_lenet.createLeNet1Model,
    Model.LSTM: tf_rnns.createLSTMModel,
    Model.GRU: tf_rnns.createGRUModel,
    Model.GRURB: tf_rnns.createGRUResetBeforeModel,
    Model.TextCNN: tf_rnns.createTextCNNModel,
}


def setRandomSeeds(seed):
    # os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)

    # os.environ['TF_DETERMINISTIC_OPS'] = '1'
    # os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    #
    # tf.config.threading.set_inter_op_parallelism_threads(1)
    # tf.config.threading.set_intra_op_parallelism_threads(1)


def getOutputPath(model: Model, language="py", lrOnly=False):
    if lrOnly and (model in [Model.VGG16, Model.ResNet20, Model.LeNet1, Model.LeNet5]):
        return OUT_LR_ONLY_PATH / model.value / language
    return OUT_PATH / model.value / language


def configGPU():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        # memory dynamic growth
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        tf.config.set_visible_devices(gpus[-1], 'GPU')
        # tf.config.set_logical_device_configuration(
        #     gpus[0],
        #     [tf.config.LogicalDeviceConfiguration(memory_limit=4096)])
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")


def CPUOrGPU():
    if tf.config.get_visible_devices('GPU'):
        return "gpu"
    return "cpu"


DatasetMap = {
    DatasetName.CIFAR: tf_vgg.loadData,
    DatasetName.MNIST: tf_lenet.loadData,
    DatasetName.IMDb: tf_rnns.loadData,
}


class TestCallback(callbacks.Callback):
    def __init__(self, test_data, train_data):
        self.test_data = test_data
        self.train_data = train_data
        self.results = {"train": [], "test": []}
        self.used_time = 0

    def on_epoch_end(self, epoch, logs={}):
        t0 = time.perf_counter()
        test_loss, test_acc = self.model.evaluate(*self.test_data, verbose=0)
        train_loss, train_acc = self.model.evaluate(*self.train_data, verbose=0)
        print(f'Epoch {epoch}, Testing error: {test_acc}, Training error: {train_acc}')
        self.results["test"].append(test_acc)
        self.results["train"].append(train_acc)
        t1 = time.perf_counter()
        self.used_time += t1 - t0


def train(model_name: Model, epochs: int, run_num: int, sgdLrOnly: bool):
    device = CPUOrGPU()
    dataset_name = Model2DatasetName[model_name]
    print(f"training for {dataset_name} x {model_name}")
    out_path = getOutputPath(model_name, lrOnly=sgdLrOnly)
    print(out_path)
    out_path.mkdir(parents=True, exist_ok=True)
    x_train, y_train, x_test, y_test, batch_size = DatasetMap[dataset_name]()
    for i in range(run_num):
        seed = SEEDS[i]
        setRandomSeeds(seed)
        print("seed:", seed)
        my_callback = TestCallback((x_test, y_test), (x_train, y_train))
        model, opt, loss_func = ModelMap[model_name](x_train, sgdLrOnly)
        model.compile(
            optimizer=opt, loss=loss_func, metrics=['accuracy'],
        )
        model.summary()

        t0 = time.perf_counter()
        history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, callbacks=[my_callback], verbose=0)
        t1 = time.perf_counter()

        total_time = t1 - t0
        print(f"total_time: {total_time}, my_callback.used_time: {my_callback.used_time}")
        training_time = total_time - my_callback.used_time
        print(f"training_time: {training_time}")

        training_errors, testing_errors = my_callback.results["train"], my_callback.results["test"]
        np.savetxt(out_path / f"training_errors_{i}.txt", training_errors)
        np.savetxt(out_path / f"testing_errors_{i}.txt", testing_errors)
        file_name_prefix = f"{dataset_name.value}-{model_name.value}"
        model.save(out_path / f'{file_name_prefix}_{i}')
        model.save_weights(out_path / f'{file_name_prefix}_weights_{i}')
        model.save_weights(out_path / f'{file_name_prefix}_weights_{i}.h5')

        test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
        train_loss, train_acc = model.evaluate(x_train, y_train, verbose=0)
        results = {
            "training_time": training_time,
            "total_time": total_time,
            "eval_time": my_callback.used_time,
            "test_loss": test_loss,
            "test_acc": test_acc,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "seed": int(seed),
        }
        print(results)
        with open(out_path / f"./time_cost_{device}_{i}.json", "w") as f:
            json.dump(results, f)


def evalProf(model, x_test, y_test, batch_size, runNum=5):
    temp_times = []
    accs = []
    for _ in range(runNum):
        t0 = time.perf_counter()
        _, acc = model.evaluate(x_test, y_test, verbose=0, batch_size=batch_size)
        t1 = time.perf_counter()
        temp_times.append(t1 - t0)
        accs.append(acc)
    for acc in accs:
        assert np.allclose(accs[0], acc), f"{accs}"
    return np.mean(temp_times[1:]), accs


def isAccuracyEqual(acc1, acc2, totalNum):
    return round(acc1 * totalNum) == round(acc2 * totalNum)


class DeployMode(Enum):
    States = "states"
    Serialization = "serialization"


def deploy(mode: DeployMode, model_name: Model, modelNum: int, sgdLrOnly: bool):
    dataset_name = Model2DatasetName[model_name]
    out_path = getOutputPath(model_name, lrOnly=sgdLrOnly)
    out_path.mkdir(parents=True, exist_ok=True)
    device = CPUOrGPU()
    eval_results = {
        "test": [],
        "test_average": None,
        "train": [],
        "train_average": None,
        "testset_same_acc": [],
        "trainset_same_acc": [],
    }
    for i in range(0, modelNum):
        print(f"verifying the {i}th model in {mode.value} mode")
        x_train, y_train, x_test, y_test, batch_size = DatasetMap[dataset_name]()
        if mode == DeployMode.Serialization:
            model = tf.keras.models.load_model(out_path / f'{dataset_name.value}-{model_name.value}_{i}')
        elif mode == DeployMode.States:
            model, opt, loss_func = ModelMap[model_name](x_train, sgdLrOnly)
            model.load_weights(out_path / f"{dataset_name.value}-{model_name.value}_weights_{i}")
            model.compile(
                optimizer=opt, loss=loss_func, metrics=['accuracy'],
            )
        else:
            raise ValueError("statesOrSerialization must be either 'states' or 'serialization'")
        model.summary()
        testing_err_gt = np.loadtxt(out_path / f"testing_errors_{i}.txt")[-1]
        training_err_gt = np.loadtxt(out_path / f"training_errors_{i}.txt")[-1]

        _, test_acc0 = model.evaluate(x_test, y_test, verbose=0, batch_size=batch_size)
        test_eval_time, test_accs = evalProf(model, x_test, y_test, batch_size)
        for acc in test_accs:
            assert np.allclose(test_acc0, acc), f"test_acc0: {test_acc0}, test_accs: {test_accs}"
        eval_results["test"].append(test_eval_time)
        eval_results["testset_same_acc"].append([
            isAccuracyEqual(test_acc0, testing_err_gt, y_test.shape[0]), test_acc0, testing_err_gt
        ])

        _, train_acc0 = model.evaluate(x_train, y_train, verbose=0, batch_size=batch_size)
        train_eval_time, train_accs = evalProf(model, x_train, y_train, batch_size)
        for acc in train_accs:
            assert np.allclose(train_acc0, acc), f"train_acc0: {train_acc0}, train_accs: {train_accs}"
        eval_results["train"].append(train_eval_time)
        eval_results["trainset_same_acc"].append([
            isAccuracyEqual(train_acc0, training_err_gt, y_train.shape[0]), train_acc0, training_err_gt
        ])

    eval_results["test_average"] = np.mean(eval_results["test"])
    eval_results["train_average"] = np.mean(eval_results["train"])
    with open(out_path / f"./deploy_eval_{device}_{mode.value}.json", "w") as f:
        json.dump(eval_results, f)


def predict(mode: DeployMode, model_name: Model, modelNum: int, sgdLrOnly: bool):
    dataset_name = Model2DatasetName[model_name]
    out_path = getOutputPath(model_name, lrOnly=sgdLrOnly)
    out_path.mkdir(parents=True, exist_ok=True)
    device = CPUOrGPU()
    for i in range(0, modelNum):
        x_train, y_train, x_test, y_test, batch_size = DatasetMap[dataset_name]()
        if mode == DeployMode.Serialization:
            model = tf.keras.models.load_model(out_path / f'{dataset_name.value}-{model_name.value}_{i}')
        elif mode == DeployMode.States:
            model, opt, loss_func = ModelMap[model_name](x_train, sgdLrOnly)
            model.load_weights(out_path / f"{dataset_name.value}-{model_name.value}_weights_{i}")
            model.compile(
                optimizer=opt, loss=loss_func, metrics=['accuracy'],
            )
        else:
            raise ValueError("statesOrSerialization must be either 'states' or 'serialization'")
        model.summary()

        y_pred = model.predict(x_test)

        if (y_pred.shape[1] == 1):
            y_pred = tf.round(y_pred).numpy().astype(np.float).reshape(-1)
        else:
            y_pred = tf.math.argmax(y_pred, 1).numpy().astype(np.uint8)
        print(y_pred.shape, y_pred, out_path.absolute())
        np.savetxt(out_path / f"predict_{device}_{mode.value}_{i}.txt", y_pred, fmt="%d")
    # eval_results["test_average"] = np.mean(eval_results["test"])
    # eval_results["train_average"] = np.mean(eval_results["train"])
    # with open(out_path / f"./deploy_eval_{device}_{mode.value}.json", "w") as f:
    #     json.dump(eval_results, f)



def main():
    parser = argparse.ArgumentParser(description='TensorFlow')
    parser.add_argument(
        'mode', type=str, choices=["train", "prof", "deploy", "pred"],
        help='train - for training\n'
             'deploy - deploy model\n'
    )
    parser.add_argument(
        'device', type=str, choices=["cpu", "CPU", "gpu", "GPU"],
        help='CPU or GPU')
    parser.add_argument(
        'model', type=str, choices=["lenet1", "lenet5", "resnet20", "vgg16", "lstm", "gru", "grurb", "textcnn"],
        help='model must be one of ["lenet1", "lenet5", "resnet20", "vgg16", "lstm", "gru", "grurb", "textcnn"]')
    parser.add_argument('epochs', type=int, nargs='?', default=100)
    parser.add_argument('run_num', type=int, nargs='?', default=5)
    parser.add_argument('gpu_device_index', type=int, nargs='?', default=0)
    parser.add_argument(
        '--sgd_lr_only', dest='sgd_lr_only', action='store_const',
        const=True, default=False,
        help='SGD optimizer with only learning rate (default: enable momentum and weight decay)'
    )
    args = parser.parse_args()
    print(args.mode, args.model, args.device, f"sgd_lr_only? {args.sgd_lr_only}")
    model = Model(args.model.lower())

    physical_devices = tf.config.list_physical_devices('GPU')
    print(f"physical_devices: {physical_devices}")
    if args.device in ["cpu", "CPU"]:
        # disable GPU
        tf.config.set_visible_devices([], 'GPU')
        assert CPUOrGPU() == "cpu"
    else:
        assert physical_devices, "cannot find any GPU device"
        # enable GPU
        tf.config.set_visible_devices([physical_devices[args.gpu_device_index]], 'GPU')
        configGPU()
        assert CPUOrGPU() == "gpu"

    if args.mode == "train":
        train(model, args.epochs, args.run_num, args.sgd_lr_only)
    elif args.mode == "deploy":
        deploy(DeployMode.States, model, args.run_num, args.sgd_lr_only)
        deploy(DeployMode.Serialization, model, args.run_num, args.sgd_lr_only)
    elif args.mode == "pred":
        predict(DeployMode.States, model, args.run_num, args.sgd_lr_only)
        predict(DeployMode.Serialization, model, args.run_num, args.sgd_lr_only)
    else:
        raise ValueError(f"Invalid mode: {args.mode}")


if __name__ == '__main__':
    main()
