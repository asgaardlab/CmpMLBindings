#!/usr/bin/env python
# -*- coding:utf-8 -*-
__date__ = '2022.01.20'

import torch.nn as nn


def createLeNet5():
    return nn.Sequential(         # nn.Sequentila allows multiple layers to stack together
            nn.Conv2d(1, 6, 5, padding=2),      # (N,1,28,28) -> (N,6,28,28)
            nn.Tanh(),
            # nn.AvgPool2d(2, stride=2),           # (N,6,28,28) -> (N,6,14,14)
            nn.MaxPool2d(2, stride=2),           # (N,6,28,28) -> (N,6,14,14)

            nn.Conv2d(6, 16, 5),                  # (N,6,14,14) -> (N,16,10,10)
            nn.Tanh(),
            # nn.AvgPool2d(2, stride=2),           # (N,16,10,10) -> (N,16,5,5)
            nn.MaxPool2d(2, stride=2),           # (N,16,10,10) -> (N,16,5,5)

            nn.Conv2d(16, 120, 5),              # (N,16,5,5) -> (N,120,1,1)
            nn.Tanh(),

            nn.Flatten(),                       # (N,120,1,1) -> (N,120)

            nn.Linear(120, 84),
            nn.Tanh(),
            nn.Linear(84, 10)
    )


def createLeNet1():
    return nn.Sequential(         # nn.Sequentila allows multiple layers to stack together
            nn.Conv2d(1, 4, 5, padding=2),      # (N,1,28,28) -> (N,6,28,28)
            nn.Tanh(),
            # nn.AvgPool2d(2, stride=2),           # (N,6,28,28) -> (N,6,14,14)
            nn.MaxPool2d(2, stride=2),           # (N,6,28,28) -> (N,6,14,14)

            nn.Conv2d(4, 12, 5),                  # (N,6,14,14) -> (N,16,10,10)
            nn.Tanh(),
            # nn.AvgPool2d(2, stride=2),           # (N,16,10,10) -> (N,16,5,5)
            nn.MaxPool2d(2, stride=2),           # (N,16,10,10) -> (N,16,5,5)

            nn.Flatten(),                       # (N,120,1,1) -> (N,120)

            nn.Linear(300, 10),
    )
