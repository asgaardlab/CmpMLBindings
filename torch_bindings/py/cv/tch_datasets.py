#!/usr/bin/env python
# -*- coding:utf-8 -*-
__date__ = '2022.01.20'


import argparse
import pathlib
import torch
import torchvision
import torchvision.transforms as tr
import torch.nn as nn
from torch.utils.data import random_split, DataLoader, Dataset, TensorDataset
import torch.optim as optim
import time
import json
import numpy as np


DATA_PATH = pathlib.Path("../../../data")
IMDb_PATH = DATA_PATH / "imdb"


def loadCIFAR10(batch_size):
    transform = tr.Compose([
        tr.ToTensor(),
        # tr.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    trainset = torchvision.datasets.CIFAR10(
        root=DATA_PATH, train=True, download=True, transform=transform
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=False
    )

    testset = torchvision.datasets.CIFAR10(
        root=DATA_PATH, train=False, download=True, transform=transform
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False
    )

    classes = (
        'plane', 'car', 'bird', 'cat', 'deer',
        'dog', 'frog', 'horse', 'ship', 'truck'
    )
    return trainloader, testloader, classes


def loadMNIST(batch_size):
    # value range [0.0 - 1.0]
    trainset = torchvision.datasets.MNIST(
        root=DATA_PATH, train=True, download=True, transform=tr.ToTensor()
    )
    testset = torchvision.datasets.MNIST(
        root=DATA_PATH, train=False, download=True, transform=tr.ToTensor()
    )

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=False
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False
    )

    classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')
    return trainloader, testloader, classes


def loadIMDb(batch_size):
    x_test = np.loadtxt(IMDb_PATH / "x_test.txt", dtype=np.int32)
    y_test = np.loadtxt(IMDb_PATH / "y_test.txt", dtype=np.float).reshape(-1, 1)
    x_train = np.loadtxt(IMDb_PATH / "x_train.txt", dtype=np.int32)
    y_train = np.loadtxt(IMDb_PATH / "y_train.txt", dtype=np.float).reshape(-1, 1)

    trainset = TensorDataset(torch.tensor(x_train, dtype=torch.int32), torch.tensor(y_train, dtype=torch.float))
    testset = TensorDataset(torch.tensor(x_test, dtype=torch.int32), torch.tensor(y_test, dtype=torch.float))
    trainloader = DataLoader(trainset, batch_size=batch_size)
    testloader = DataLoader(testset, batch_size=batch_size)

    classes = ('0', '1')
    return trainloader, testloader, classes
