#!/usr/bin/env python
# -*- coding:utf-8 -*-
__date__ = '2022.01.20'

import torch.nn as nn


def conv_3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)


def conv_1x1(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=False):
        super(Block, self).__init__()
        self.downsample = downsample

        if self.downsample:
            self.conv1 = conv_3x3(in_channels, out_channels, stride=2)
        else:
            self.conv1 = conv_3x3(in_channels, out_channels)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = conv_3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = None
        self.bn3 = None
        if self.downsample:
            self.conv3 = conv_1x1(in_channels, out_channels, stride=2)
            self.bn3 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample:
            identity = self.conv3(x)
            identity = self.bn3(identity)
        # print("identity:", identity.shape)
        # print("out:", out.shape)
        out += identity
        out = self.relu(out)

        return out


class ResNet20(nn.Module):
    def __init__(self, in_channels=3, num_classes=10):
        super(ResNet20, self).__init__()

        self.conv_1 = conv_3x3(in_channels, 16, 1)
        self.bn_1 = nn.BatchNorm2d(16)
        self.relu_1 = nn.ReLU()

        self.stack1 = self._create_stack(16, 16, downsample_first=False)
        self.stack2 = self._create_stack(16, 32, downsample_first=True)
        self.stack3 = self._create_stack(32, 64, downsample_first=True)

        self.avg_last = nn.AvgPool2d((8, 8))
        self.relu_last = nn.ReLU()
        self.out = nn.Linear(64, num_classes)

    def forward(self, x):
        # print(x.shape)
        x11 = self.conv_1(x)
        # print(x11.shape)
        x12 = self.bn_1(x11)
        # print(x12.shape)
        x13 = self.relu_1(x12)

        # print("x13:", x13.shape)
        # s1 = self.forward_stack(self.stack1, x13)
        s1 = self.stack1(x13)
        # print("s1:", s1.shape)
        # s2 = self.forward_stack(self.stack2, s1)
        s2 = self.stack2(s1)
        # print("s2:", s2.shape)
        # s3 = self.forward_stack(self.stack3, s2)
        s3 = self.stack3(s2)
        # print("s3:", s3.shape)

        s3_avg = self.avg_last(s3)
        ro = self.relu_last(s3_avg)
        ro = ro.reshape(ro.shape[0], -1)
        out = self.out(ro)
        return out

    def forward_stack(self, stack, x):
        is_downsample = stack[0]
        downsample = None
        if is_downsample:
            blocks = stack[1:-1]
            downsample = stack[-1]
            # print(downsample)
        else:
            blocks = stack[1:]
        input = None
        first = True
        for block in blocks:
            layer0, layer1 = block
            # print(layer0)
            # print(layer1)
            if first:
                identical = x
                o1 = layer0[2](layer0[1](layer0[0](x)))
                first = False
            else:
                identical = input
                o1 = layer0[2](layer0[1](layer0[0](input)))
            o2 = layer1[1](layer1[0](o1))
            # print("o2:", o2.shape)
            # print("identical:", identical.shape)
            if o2.shape != identical.shape:
                identical = downsample[1](downsample[0](identical))
                # print("identical downsample:", identical.shape)
            input = layer1[2](identical + o2)
        return input

    def _create_stack(self, in_channels, out_channels, downsample_first=False):
        b1 = Block(in_channels, out_channels, downsample_first)
        b2 = Block(out_channels, out_channels)
        b3 = Block(out_channels, out_channels)
        return nn.Sequential(b1, b2, b3)
