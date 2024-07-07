#!/usr/bin/env python
# -*- coding:utf-8 -*-
__date__ = '2022.01.20'

import collections

import torch.nn as nn

VGG16_CONFIG = [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"]


class VGG16(nn.Module):
    def __init__(self, in_channels=3, num_classes=10):
        super(VGG16, self).__init__()
        self.in_channels = in_channels

        self.cnn_model = self._create_conv_layers()

        self.fcs = nn.Sequential(
            collections.OrderedDict(
                [
                    ("flatten", nn.Flatten()),
                    ("fc_linear_1", nn.Linear(512 * 1 * 1, 4096)),
                    ("fc_relu_1", nn.ReLU()),
                    ("fc_dropout_1", nn.Dropout(p=0.5)),
                    ("fc_linear_2", nn.Linear(4096, 4096)),
                    ("fc_relu_2", nn.ReLU()),
                    ("fc_dropout_2", nn.Dropout(p=0.5)),
                    ("fc_out", nn.Linear(4096, num_classes)),
                ]
            )
        )

    def forward(self, x):
        # print(x.shape)
        x = self.cnn_model(x)
        x = self.fcs(x)
        return x

    def _create_conv_layers(self):
        layers = collections.OrderedDict()
        in_channels = self.in_channels

        for i, x in enumerate(VGG16_CONFIG):
            if type(x) == int:
                out_channels = x
                print(f"config for {x}: in {in_channels}, out {out_channels}")
                layers[f"conv2d-{i}a"] = nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=(3, 3),
                    stride=(1, 1),
                    padding=(1, 1),
                    bias=False
                )
                layers[f"bnrm2d-{i}a"] = nn.BatchNorm2d(x)
                layers[f"relu-{i}b"] = nn.ReLU()
                in_channels = x
            elif x == "M":
                layers[f"maxpool2d-{i}a"] = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        return nn.Sequential(layers)
