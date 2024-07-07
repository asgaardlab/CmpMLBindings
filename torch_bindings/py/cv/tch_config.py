#!/usr/bin/env python
# -*- coding:utf-8 -*-
__date__ = '2022.01.20'

import torch


DEVICE = None
DEVICE_NAME = None
DEVICE_NAME_STR = None


def setDeviceToCPU():
    global DEVICE_NAME, DEVICE, DEVICE_NAME_STR
    DEVICE_NAME_STR = DEVICE_NAME = "cpu"
    DEVICE = torch.device(DEVICE_NAME)
    print(f"Set device to {DEVICE_NAME}")


def setDeviceToGPU(device_index=0):
    global DEVICE_NAME, DEVICE, DEVICE_NAME_STR
    assert torch.cuda.is_available(), "CUDA is unavailable!"
    DEVICE_NAME = f"cuda:{device_index}" if torch.cuda.is_available() else "cpu"
    DEVICE_NAME_STR = "gpu"
    DEVICE = torch.device(DEVICE_NAME)
    print(f"Set device to {DEVICE_NAME}")
