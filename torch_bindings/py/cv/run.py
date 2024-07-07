#!/usr/bin/env python
# -*- coding:utf-8 -*-
__date__ = '2022.01.13'

import argparse
import json
import pathlib
import time
from enum import Enum
import random

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchinfo

import tch_config
import tch_datasets
import tch_lenet
import tch_rnns
import tch_resnet
import tch_vgg

OUT_PATH = pathlib.Path("../../../out/pytorch")
OUT_LR_ONLY_PATH = pathlib.Path("../../../out/pytorch_lr_only")
TEST_SET_SIZE = 10000
TRAIN_SET_SIZE = 60000
SEEDS = np.loadtxt("../../../random_seeds.txt").astype(np.uint32)


class DatasetName(Enum):
    CIFAR = "cifar"
    MNIST = "mnist"
    IMDb = "imdb"


BATCH_SIZE_MAP = {
    DatasetName.CIFAR: 128,
    DatasetName.MNIST: 128,
    DatasetName.IMDb: 256,
}

DatasetMap = {
    DatasetName.CIFAR: tch_datasets.loadCIFAR10,
    DatasetName.MNIST: tch_datasets.loadMNIST,
    DatasetName.IMDb: tch_datasets.loadIMDb,
}

LOSS_FUNC = {
    DatasetName.CIFAR: nn.CrossEntropyLoss(),
    DatasetName.MNIST: nn.CrossEntropyLoss(),
    DatasetName.IMDb: nn.BCELoss(),
}


class Model(Enum):
    VGG16 = "vgg16"
    ResNet20 = "resnet20"
    LeNet1 = "lenet1"
    LeNet5 = "lenet5"
    GRU = "gru"
    LSTM = "lstm"
    TextCNN = "textcnn"


def setRandomSeeds(seed):
    # os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def createVGG16(sgdLrOnly=False):
    net = tch_vgg.VGG16()
    if sgdLrOnly:
        opt = optim.SGD(net.parameters(), lr=5e-2)
    else:
        opt = optim.SGD(net.parameters(), momentum=9e-1, lr=1e-1, weight_decay=1e-4)
    return net, opt


def createResNet20(sgdLrOnly=False):
    net = tch_resnet.ResNet20()
    if sgdLrOnly:
        opt = optim.SGD(net.parameters(), lr=5e-2)
    else:
        opt = optim.SGD(net.parameters(), momentum=9e-1, lr=1e-1, weight_decay=1e-4)
    return net, opt


def createLeNet5(sgdLrOnly=False):
    net = tch_lenet.createLeNet5()
    if sgdLrOnly:
        opt = optim.SGD(net.parameters(), lr=5e-2)
    else:
        opt = optim.SGD(net.parameters(), momentum=9e-1, lr=5e-2)
    return net, opt


def createLeNet1(sgdLrOnly=False):
    net = tch_lenet.createLeNet1()
    if sgdLrOnly:
        opt = optim.SGD(net.parameters(), lr=5e-2)
    else:
        opt = optim.SGD(net.parameters(), momentum=9e-1, lr=5e-2)
    return net, opt


def createLSTM(_):
    net = tch_rnns.LSTM()
    opt = optim.Adam(net.parameters(), lr=8e-5)
    return net, opt


def createGRU(_):
    net = tch_rnns.GRU()
    opt = optim.Adam(net.parameters(), lr=3e-4)
    return net, opt


def createTextCNN(_):
    net = tch_rnns.TextCNN()
    opt = optim.Adam(net.parameters(), lr=8e-5)
    return net, opt


ModelMap = {
    Model.VGG16: createVGG16,
    Model.ResNet20: createResNet20,
    Model.LeNet1: createLeNet1,
    Model.LeNet5: createLeNet5,
    Model.LSTM: createLSTM,
    Model.GRU: createGRU,
    Model.TextCNN: createTextCNN,
}

Model2DatasetName = {
    Model.VGG16: DatasetName.CIFAR,
    Model.ResNet20: DatasetName.CIFAR,
    Model.LeNet1: DatasetName.MNIST,
    Model.LeNet5: DatasetName.MNIST,
    Model.GRU: DatasetName.IMDb,
    Model.LSTM: DatasetName.IMDb,
    Model.TextCNN: DatasetName.IMDb,
}

DatasetShape = {
    DatasetName.CIFAR: (BATCH_SIZE_MAP[DatasetName.CIFAR], 3, 32, 32),
    DatasetName.MNIST: (BATCH_SIZE_MAP[DatasetName.MNIST], 1, 28, 28),
    DatasetName.IMDb: (BATCH_SIZE_MAP[DatasetName.IMDb], tch_rnns.EMBEDDING_VEC_LEN),
}

DatasetDTypes = {
    DatasetName.CIFAR: None,
    DatasetName.MNIST: None,
    DatasetName.IMDb: [torch.int32]*len(DatasetShape[DatasetName.IMDb]),
}

IsBinaryEval = {
    DatasetName.CIFAR: False,
    DatasetName.MNIST: False,
    DatasetName.IMDb: True,
}


def loadDataset(name: DatasetName):
    batch_size = BATCH_SIZE_MAP[name]
    trainloader, testloader, classes = DatasetMap[name](batch_size)
    return trainloader, testloader, classes


def getOutputPath(model: Model, language="py", lrOnly=False):
    if lrOnly and (model in [Model.VGG16, Model.ResNet20, Model.LeNet1, Model.LeNet5]):
        return OUT_LR_ONLY_PATH / model.value / language
    return OUT_PATH / model.value / language


def evaluation(net, dataloader, binary=False):
    net.eval()
    total, correct = 0, 0
    for data in dataloader:
        inputs, labels = data
        inputs, labels = inputs.to(tch_config.DEVICE), labels.to(tch_config.DEVICE)
        output = net(inputs)
        if binary:
            pred = torch.round(output)
        else:
            max_pred, pred = torch.max(output.data, dim=1)
        total += labels.size(0)
        correct += torch.sum(pred == labels).item()
    return correct / total


def evalProf(net, dataloader, binary, runNum=5):
    temp_times = []
    accs = []
    for _ in range(runNum):
        t0 = time.perf_counter()
        acc = evaluation(net, dataloader, binary)
        t1 = time.perf_counter()
        temp_times.append(t1 - t0)
        accs.append(acc)
    return np.mean(temp_times[1:]), accs


def run(model: Model, epochs: int, run_num: int, sgdLrOnly: bool):
    dataset_name = Model2DatasetName[model]
    print(f"training for {dataset_name} x {model}")
    out_path = getOutputPath(model, lrOnly=sgdLrOnly)
    out_path.mkdir(parents=True, exist_ok=True)
    for i in range(run_num):
        seed = int(SEEDS[i])
        setRandomSeeds(seed)
        print(f"running {i} / {run_num}, total epochs: {epochs}, seed: {seed}")

        # load dataset
        trainloader, testloader, classes = loadDataset(dataset_name)

        # setup model and optimizer
        net, opt = ModelMap[model](sgdLrOnly)
        torch.save(net.state_dict(), out_path / f'{dataset_name.value}-{model.value}_{i}_init.pth')

        # summary of model
        net.to(tch_config.DEVICE)
        print(torchinfo.summary(net, DatasetShape[dataset_name], dtypes=DatasetDTypes[dataset_name], device=tch_config.DEVICE))

        # training here
        net, train_accs, test_accs, total_time, eval_time = fit(
            net, opt, LOSS_FUNC[dataset_name], trainloader, testloader, IsBinaryEval[dataset_name], epochs=epochs
        )

        # evaluating and saving the results
        train_acc = evaluation(net, trainloader, IsBinaryEval[dataset_name])
        test_acc = evaluation(net, testloader, IsBinaryEval[dataset_name])
        training_time = total_time - eval_time
        print(training_time)
        results = {
            "training_time": training_time,
            "total_time": total_time,
            "eval_time": eval_time,
            "test_acc": test_acc,
            "train_acc": train_acc,
            "seed": int(seed),
        }
        print(results)

        with open(out_path / f"time_cost_{tch_config.DEVICE_NAME_STR}_{i}.json", "w") as f:
            json.dump(results, f)
        np.savetxt(out_path / f"training_errors_{i}.txt", train_accs)
        np.savetxt(out_path / f"testing_errors_{i}.txt", test_accs)

        net.eval()
        # saving model states
        torch.save(net.state_dict(), out_path / f'{dataset_name.value}-{model.value}_{i}.pth')

        model_scripted = torch.jit.script(net)
        model_scripted.save(out_path / f"scripted_{dataset_name.value}-{model.value}_{i}_gpu.pt")

        # data = iter(trainloader).__next__()
        # saving traced model for other bindings (serialization)
        # traced_script_module = torch.jit.trace(net.to(tch_config.DEVICE), data[0].to(tch_config.DEVICE))
        # traced_script_module.save(out_path / f"traced_{dataset_name.value}-{model.value}_{i}_gpu.pt")
        # cpu_device = torch.device('cpu')
        # traced_script_module = torch.jit.trace(net.to(cpu_device), data[0])
        # traced_script_module.save(out_path / f"traced_{dataset_name.value}-{model.value}_{i}.pt")

        # net.train()
        # model_scripted = torch.jit.script(net)
        # model_scripted.save(out_path / f"scripted_{dataset_name.value}-{model.value}_{i}_gpu_train.pt")
        cpu_device = torch.device('cpu')
        net.to(cpu_device)
        model_scripted = torch.jit.script(net)
        model_scripted.save(out_path / f"scripted_{dataset_name.value}-{model.value}_{i}.pt")

        # traced_script_module = torch.jit.trace(net.to(tch_config.DEVICE), data[0].to(tch_config.DEVICE))
        # traced_script_module.save(out_path / f"traced_{dataset_name.value}-{model.value}_{i}_gpu_train.pt")
        # cpu_device = torch.device('cpu')
        # traced_script_module = torch.jit.trace(net.to(cpu_device), data[0])
        # traced_script_module.save(out_path / f"traced_{dataset_name.value}-{model.value}_{i}_train.pt")


def fit(net, opt, loss_fn, trainloader, testloader, binary, epochs=16):
    # And mark rest of the values as zeros.
    train_accs = []
    test_accs = []
    t0 = time.perf_counter()
    eval_time_used = 0.0
    for epoch in range(epochs):
        net.train()
        for i, data in enumerate(trainloader):  # Iterating through the train loader
            inputs, labels = data
            inputs, labels = inputs.to(tch_config.DEVICE), labels.to(tch_config.DEVICE)
            opt.zero_grad()  # Reset the gradient in every iteration
            outputs = net(inputs)
            loss = loss_fn(outputs, labels)  # Loss forward pass
            loss.backward()  # Loss backaed pass
            opt.step()  # Update all the parameters by the given learnig rule
        t_eval = time.perf_counter()
        train_acc = evaluation(net, trainloader, binary)
        test_acc = evaluation(net, testloader, binary)
        train_accs.append(train_acc)
        test_accs.append(test_acc)
        print(f"epoch {epoch} - train_acc: {train_acc}, test_acc: {test_acc}")
        eval_time_used += (time.perf_counter() - t_eval)
    total_time_used = time.perf_counter() - t0
    return net, train_accs, test_accs, total_time_used, eval_time_used


def isAccuracyEqual(acc1, acc2, totalNum):
    return round(acc1 * totalNum) == round(acc2 * totalNum)


class DeployMode(Enum):
    States = "states"
    Serialization = "serialization"


def deployEvaluation(mode: DeployMode, model: Model, modelNum: int, sgdLrOnly: bool, trainMode=False, scripted=True):
    dataset_name = Model2DatasetName[model]
    out_path = getOutputPath(model, lrOnly=sgdLrOnly)
    eval_results = {
        "test": [],
        "test_average": None,
        "train": [],
        "train_average": None,
        "testset_same_acc": [],
        "trainset_same_acc": [],
        "test_acc_stable": [],
        "train_acc_stable": [],
    }
    prefix_name = "scripted" if scripted else "traced"

    for i in range(0, modelNum):
        # seed = SEEDS[i]
        # setRandomSeeds(seed)
        print(f"verifying the {i}th model in {mode.value} mode")
        trainloader, testloader, classes = loadDataset(dataset_name)

        end_str = ""
        if mode == DeployMode.States:
            net, _ = ModelMap[model](sgdLrOnly)
            states = torch.load(
                out_path / f"{dataset_name.value}-{model.value}_{i}.pth",
                map_location=tch_config.DEVICE
            )
            net.load_state_dict(states)
        elif mode == DeployMode.Serialization:
            if trainMode:
                end_str = "_train"
            gpu_model_path = out_path / f"{prefix_name}_{dataset_name.value}-{model.value}_{i}_gpu{end_str}.pt"
            if gpu_model_path.exists() and tch_config.DEVICE_NAME_STR == "gpu":
                net = torch.jit.load(gpu_model_path)
            else:
                net = torch.jit.load(out_path / f"{prefix_name}_{dataset_name.value}-{model.value}_{i}{end_str}.pt")
        else:
            raise NotImplemented("cannot reach")
        net.to(tch_config.DEVICE)
        testing_err_gt = np.loadtxt(out_path / f"testing_errors_{i}.txt")[-1]
        training_err_gt = np.loadtxt(out_path / f"training_errors_{i}.txt")[-1]

        test_acc = evaluation(net, testloader, IsBinaryEval[dataset_name])
        test_eval_time, test_accs = evalProf(net, testloader, IsBinaryEval[dataset_name])
        test_acc_stable = True
        for acc in test_accs:
            if acc != test_acc:
                test_acc_stable = False
                break
        eval_results["test"].append(test_eval_time)
        eval_results["test_acc_stable"].append(test_acc_stable)

        train_acc = evaluation(net, trainloader, IsBinaryEval[dataset_name])
        train_eval_time, train_accs = evalProf(net, trainloader, IsBinaryEval[dataset_name])
        train_acc_stable = True
        for acc in train_accs:
            if acc != train_acc:
                train_acc_stable = False
                break
        eval_results["train"].append(train_eval_time)
        eval_results["train_acc_stable"].append(train_acc_stable)

        eval_results["testset_same_acc"].append([
            isAccuracyEqual(test_acc, testing_err_gt, TEST_SET_SIZE), test_acc, testing_err_gt
        ])
        eval_results["trainset_same_acc"].append([
            isAccuracyEqual(train_acc, training_err_gt, TRAIN_SET_SIZE), train_acc, training_err_gt
        ])

    eval_results["test_average"] = np.mean(eval_results["test"])
    eval_results["train_average"] = np.mean(eval_results["train"])
    with open(out_path / f"./deploy_eval_{tch_config.DEVICE_NAME_STR}_{mode.value}{end_str}_{prefix_name}.json", "w") as f:
        json.dump(eval_results, f)


def predict(mode: DeployMode, model: Model, modelNum: int, sgdLrOnly: bool, trainMode=False, scripted=True):
    dataset_name = Model2DatasetName[model]
    out_path = getOutputPath(model, lrOnly=sgdLrOnly)
    prefix_name = "scripted" if scripted else "traced"
    for i in range(0, modelNum):
        # seed = SEEDS[i]
        # setRandomSeeds(seed)
        print(f"verifying the {i}th model in {mode.value} mode")
        trainloader, testloader, classes = loadDataset(dataset_name)

        end_str = ""
        if mode == DeployMode.States:
            net, _ = ModelMap[model](sgdLrOnly)
            states = torch.load(
                out_path / f"{dataset_name.value}-{model.value}_{i}.pth",
                map_location=tch_config.DEVICE
            )
            net.load_state_dict(states)
        elif mode == DeployMode.Serialization:
            if trainMode:
                end_str = "_train"
            gpu_model_path = out_path / f"{prefix_name}_{dataset_name.value}-{model.value}_{i}_gpu{end_str}.pt"
            if gpu_model_path.exists() and tch_config.DEVICE_NAME_STR == "gpu":
                net = torch.jit.load(gpu_model_path)
            else:
                net = torch.jit.load(out_path / f"{prefix_name}_{dataset_name.value}-{model.value}_{i}{end_str}.pt")
        else:
            raise NotImplemented("cannot reach")
        net.to(tch_config.DEVICE)

        net.eval()
        preds = []
        for data in testloader:
            inputs, labels = data
            inputs, labels = inputs.to(tch_config.DEVICE), labels.to(tch_config.DEVICE)
            output = net(inputs)
            if IsBinaryEval[dataset_name]:
                pred = torch.round(output)
            else:
                max_pred, pred = torch.max(output.data, dim=1)
            preds.extend(pred.tolist())
        preds = np.array(preds).astype(np.uint8).reshape(-1)
        print(preds.shape, preds, out_path.absolute())
        np.savetxt(out_path / f"predict_{tch_config.DEVICE_NAME_STR}_{mode.value}_{i}.txt", preds, fmt="%d")


def main():
    parser = argparse.ArgumentParser(description="PyTorch's bindings")
    parser.add_argument(
        'mode', type=str, choices=["train", "deploy", "pred"],
        help='train - for training\n'
             'deploy - deploy model\n'
    )
    parser.add_argument(
        'device', type=str, choices=["cpu", "CPU", "gpu", "GPU"],
        help='CPU or GPU')
    parser.add_argument(
        'model', type=str, choices=["lenet5", "lenet1", "vgg16", "resnet20", "lstm", "gru", "textcnn"],
        help='model must be one of ["lenet5", "lenet1", "vgg16", "resnet20", "lstm", "gru", "textcnn"]')
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

    if args.device.lower() == "cpu":
        tch_config.setDeviceToCPU()
    else:
        tch_config.setDeviceToGPU(args.gpu_device_index)

    if args.mode == "train":
        run(model, args.epochs, args.run_num, args.sgd_lr_only)
    elif args.mode == "deploy":
        # deployEvaluation(DeployMode.Serialization, model, args.run_num, args.sgd_lr_only, trainMode=True)
        deployEvaluation(DeployMode.States, model, args.run_num, args.sgd_lr_only)
        deployEvaluation(DeployMode.Serialization, model, args.run_num, args.sgd_lr_only)
        # deployEvaluation(DeployMode.Serialization, model, args.run_num, args.sgd_lr_only, trainMode=True, scripted=False)
        # deployEvaluation(DeployMode.Serialization, model, args.run_num, args.sgd_lr_only, scripted=False)
    elif args.mode == "pred":
        predict(DeployMode.States, model, args.run_num, args.sgd_lr_only)
        predict(DeployMode.Serialization, model, args.run_num, args.sgd_lr_only)
    else:
        raise ValueError(f"Invalid mode: {args.mode}")


if __name__ == '__main__':
    main()
