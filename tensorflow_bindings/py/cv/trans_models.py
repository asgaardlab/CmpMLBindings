#!/usr/bin/env python
# -*- coding:utf-8 -*-
__date__ = '2022.01.20'

import argparse
import pathlib
import tensorflow as tf
import tensorflowjs as tfjs

import run


OUT_PATH = pathlib.Path("../../../out/tensorflow")
OUT_LR_ONLY_PATH = pathlib.Path("../../../out/tensorflow_lr_only")


def trainsModelToJS(model_name, sgdLrOnly=False):
    dataset_name = run.Model2DatasetName[model_name]
    if sgdLrOnly:
        model_path = OUT_LR_ONLY_PATH / f"{model_name.value}/py"
    else:
        model_path = OUT_PATH / f"{model_name.value}/py"
    for i in range(5):
        model = tf.keras.models.load_model(model_path / f"{dataset_name.value}-{model_name.value}_{i}")
        tfjs.converters.save_keras_model(model, model_path / f"{dataset_name.value}-{model_name.value}_{i}_tfjs")


def main():
    parser = argparse.ArgumentParser(description='LeNet5 on TensorFlow')
    parser.add_argument(
        'model', type=str, choices=["lenet1", "lenet5", "vgg16", "lstm", "gru", "grurb", "textcnn"],
        help='model must be one of ["lenet1", "lenet5", "vgg16", "lstm", "gru", "grurb", "textcnn"]')
    parser.add_argument(
        'language', type=str, choices=["ts"],
        help='ts')
    parser.add_argument(
        '--sgd_lr_only', dest='sgd_lr_only', action='store_const',
        const=True, default=False,
        help='SGD optimizer with only learning rate (default: enable momentum and weight decay)'
    )
    args = parser.parse_args()
    print(args.model, args.language, f"sgd_lr_only? {args.sgd_lr_only}")
    model = run.Model(args.model.lower())
    trans_map = {
        "ts": trainsModelToJS,
    }
    trans_map[args.language](model, args.sgd_lr_only)


if __name__ == '__main__':
    # showStatesInRustModel(run.Model.VGG16)
    main()
