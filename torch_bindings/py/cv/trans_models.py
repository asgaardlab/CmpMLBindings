#!/usr/bin/env python
# -*- coding:utf-8 -*-
__date__ = '2022.01.20'

import argparse
import numpy as np
import torch

import exportsd
import importsd
import run


def saveTorchScriptInTrainMode(model_name: run.Model, sgdLrOnly=False):
    print(f"saving {model_name.value} in train mode sgdLrOnly? {sgdLrOnly}")
    dataset_name = run.Model2DatasetName[model_name]
    out_path = run.getOutputPath(model_name, lrOnly=sgdLrOnly)

    model, _ = run.ModelMap[model_name](sgdLrOnly)
    trainloader, testloader, classes = run.loadDataset(dataset_name)
    data = iter(trainloader).__next__()

    cpu_device = torch.device("cpu")
    gpu_device = torch.device("cuda:0")
    for device in [cpu_device, gpu_device]:
        for i in range(5):
            print(f"loading {i}")
            states = torch.load(out_path / f"{dataset_name.value}-{model_name.value}_{i}.pth", map_location=cpu_device)
            model.load_state_dict(states)
            # print(model(data[0]).shape)

            # model.train()
            # traced_script_module = torch.jit.trace(model.to(device), data[0].to(device))

            # print(data[0].shape)
            # print(traced_script_module)
            # if device == cpu_device:
            #     traced_script_module.save(out_path / f"traced_{dataset_name.value}-{model_name.value}_{i}_train.pt")
            # else:
            #     traced_script_module.save(out_path / f"traced_{dataset_name.value}-{model_name.value}_{i}_gpu_train.pt")

            # model_scripted = torch.jit.script(model)
            # model_scripted.save(out_path / f"scripted_{dataset_name.value}-{model_name.value}_{i}_train.pt")
            model.to(device)
            model.eval()
            model_scripted = torch.jit.script(model)
            if device == cpu_device:
                model_scripted.save(out_path / f"scripted_{dataset_name.value}-{model_name.value}_{i}.pt")
            else:
                model_scripted.save(out_path / f"scripted_{dataset_name.value}-{model_name.value}_{i}_gpu.pt")


def showStatesInRustModel(model: run.Model):
    database = run.Model2DatasetName[model]
    device = torch.device('cpu')
    rs_out_path = run.getOutputPath(model, "rs")
    states = torch.load(rs_out_path / f"{database.value}-{model.value}_{0}.pth", map_location=device)
    states_dict = states.state_dict()
    keys = list(states_dict.keys())
    features_key = list(filter(lambda x: x.startswith("features"), keys))
    classifier_key = list(filter(lambda x: x.startswith("classifier"), keys))
    features_key.sort()
    classifier_key.sort()
    print("===============")
    for name in features_key:
        print(name, states_dict[name].shape)
    print("===============")
    for name in classifier_key:
        print(name, states_dict[name].shape)
    print("===============")
    keys.sort()
    if not features_key and not classifier_key:
        for name in keys:
            print(name, states_dict[name].shape)


def transStatesToRust(model: run.Model, sgdLrOnly=False, toPY=False):
    database = run.Model2DatasetName[model]
    device = torch.device('cpu')
    py_out_path = run.getOutputPath(model, lrOnly=sgdLrOnly)
    rs_out_path = run.getOutputPath(model, "rs", lrOnly=sgdLrOnly)
    first = True
    replacement_kw_rs = {
        # cnn
        "cnn_model.conv2d-0a.": "features|0|",
        "cnn_model.bnrm2d-0a.": "features|1|",

        "cnn_model.conv2d-1a.": "features|3|",
        "cnn_model.bnrm2d-1a.": "features|4|",

        "cnn_model.conv2d-3a.": "features|7|",
        "cnn_model.bnrm2d-3a.": "features|8|",

        "cnn_model.conv2d-4a.": "features|10|",
        "cnn_model.bnrm2d-4a.": "features|11|",

        "cnn_model.conv2d-6a.": "features|14|",
        "cnn_model.bnrm2d-6a.": "features|15|",

        "cnn_model.conv2d-7a.": "features|17|",
        "cnn_model.bnrm2d-7a.": "features|18|",

        "cnn_model.conv2d-8a.": "features|20|",
        "cnn_model.bnrm2d-8a.": "features|21|",

        "cnn_model.conv2d-10a.": "features|24|",
        "cnn_model.bnrm2d-10a.": "features|25|",

        "cnn_model.conv2d-11a.": "features|27|",
        "cnn_model.bnrm2d-11a.": "features|28|",

        "cnn_model.conv2d-12a.": "features|30|",
        "cnn_model.bnrm2d-12a.": "features|31|",

        "cnn_model.conv2d-14a.": "features|34|",
        "cnn_model.bnrm2d-14a.": "features|35|",

        "cnn_model.conv2d-15a.": "features|37|",
        "cnn_model.bnrm2d-15a.": "features|38|",

        "cnn_model.conv2d-16a.": "features|40|",
        "cnn_model.bnrm2d-16a.": "features|41|",
        # fcs
        "fcs.fc_linear_1.": "classifier|0|",
        "fcs.fc_linear_2.": "classifier|3|",
        "fcs.fc_out.": "classifier|6|",
    }
    replacement_map_vgg16 = {}
    for i in range(5):
        if toPY:
            states = torch.load(rs_out_path / f"{database.value}-{model.value}_{i}.pth", map_location=device).state_dict()
        else:
            states = torch.load(py_out_path / f"{database.value}-{model.value}_{i}.pth", map_location=device)

        if first:
            for name, data in states.items():
                print(name, data.shape)
            if model == run.Model.VGG16:
                py_states = torch.load(py_out_path / f"{database.value}-{model.value}_{i}.pth", map_location=device)
                for name, data in py_states.items():
                    for k, v in replacement_kw_rs.items():
                        if name.startswith(k):
                            rename_to = name.replace(k, v)
                            replacement_map_vgg16[name] = rename_to
                            print(name, "to", rename_to)
                            break
                    else:
                        print("!!!!!!!!!!!!", name)
                assert len(py_states) == len(replacement_map_vgg16)
            first = False

        if model == run.Model.LeNet5:
            replacement_map = {
                "0.bias": "bias",
                "3.bias": "bias__2",
                "6.bias": "bias__4",
                "9.bias": "bias__6",
                "11.bias": "bias__8",
                "0.weight": "weight",
                "3.weight": "weight__3",
                "6.weight": "weight__5",
                "9.weight": "weight__7",
                "11.weight": "weight__9",
            }
        elif model == run.Model.LeNet1:
            replacement_map = {
                "0.bias": "bias",
                "3.bias": "bias__2",
                "7.bias": "bias__4",
                "0.weight": "weight",
                "3.weight": "weight__3",
                "7.weight": "weight__5",
            }
        elif model == run.Model.VGG16:
            replacement_map = replacement_map_vgg16
        elif model == run.Model.LSTM or model == run.Model.GRU:
            replacement_map = {
                "embed.weight": "embedding|weight",
                "rnn_model.weight_ih_l0": "rnn_model|weight_ih_l0",
                "rnn_model.weight_hh_l0": "rnn_model|weight_hh_l0",
                "rnn_model.bias_ih_l0": "rnn_model|bias_ih_l0",
                "rnn_model.bias_hh_l0": "rnn_model|bias_hh_l0",
                "fc.weight": "dense|weight",
                "fc.bias": "dense|bias",
            }
        else:
            raise ValueError(f"wrong model name: {model}")

        original_keys = list(states.keys())
        if toPY:
            replacement_map_rev = {v: k for k, v in replacement_map.items()}
            nps = {replacement_map_rev[k]: states[k].to(device) for k in original_keys}
            torch.save(nps, rs_out_path / f"{database.value}-{model.value}_{i}_py.pth")
        else:
            nps = {replacement_map[k]: states[k] for k in original_keys}
            np.savez(py_out_path / f"{database.value}-{model.value}_{i}_rust.npz", **nps)


def transStatesToDotNet(model: run.Model, sgdLrOnly=False, toPY=False):
    database = run.Model2DatasetName[model]
    device = torch.device('cpu')
    py_out_path = run.getOutputPath(model, lrOnly=sgdLrOnly)
    dotnet_out_path = run.getOutputPath(model, "dotnet", lrOnly=sgdLrOnly)
    for i in range(5):
        if toPY:
            with open(dotnet_out_path / f"{database.value}_model_{i}.bin", "rb") as stream:
                states = importsd.load_state_dict(stream)
        else:
            states = torch.load(py_out_path / f"{database.value}-{model.value}_{i}.pth", map_location=device)
        for name, data in states.items():
            print(name, data.shape)
        original_keys = list(states.keys())

        if model == run.Model.LeNet5:
            replacement_map = {
                "0.weight": "conv1.weight",
                "0.bias": "conv1.bias",
                "3.weight": "conv2.weight",
                "3.bias": "conv2.bias",
                "6.weight": "conv3.weight",
                "6.bias": "conv3.bias",
                "9.weight": "fc1.weight",
                "9.bias": "fc1.bias",
                "11.weight": "fc2.weight",
                "11.bias": "fc2.bias",
            }
        elif model == run.Model.LeNet1:
            replacement_map = {
                "0.weight": "conv1.weight",
                "0.bias": "conv1.bias",
                "3.weight": "conv2.weight",
                "3.bias": "conv2.bias",
                "7.weight": "fc1.weight",
                "7.bias": "fc1.bias",
            }
        elif model == run.Model.VGG16:
            py_states = torch.load(py_out_path / f"{database.value}-{model.value}_{i}.pth", map_location=device)
            replacement_map = {}
            for name in py_states:
                assert isinstance(name, str)
                if name.startswith("cnn_model"):
                    to_name = name.replace("cnn_model", "layers")
                elif name.startswith("fcs"):
                    # if "fc_linear_1.weight" in states:
                    #     to_name = name.replace("fcs.", "")
                    # else:
                    #     to_name = name.replace("fcs", "layers")
                    to_name = name.replace("fcs", "layers")
                    # to_name = name.replace("fcs.", "")
                replacement_map[name] = to_name
        elif model == run.Model.LSTM:
            replacement_map = {
                "embed.weight": "embedding.weight",
                "rnn_model.weight_ih_l0": "lstm.weight_ih_l0",
                "rnn_model.weight_hh_l0": "lstm.weight_hh_l0",
                "rnn_model.bias_ih_l0": "lstm.bias_ih_l0",
                "rnn_model.bias_hh_l0": "lstm.bias_hh_l0",
                # "fc.weight": "dense.weight",
                # "fc.bias": "dense.bias",
                "fc.weight": "layers.dense.weight",
                "fc.bias": "layers.dense.bias",
            }
        elif model == run.Model.GRU:
            replacement_map = {
                "embed.weight": "embedding.weight",
                "rnn_model.weight_ih_l0": "gru.weight_ih_l0",
                "rnn_model.weight_hh_l0": "gru.weight_hh_l0",
                "rnn_model.bias_ih_l0": "gru.bias_ih_l0",
                "rnn_model.bias_hh_l0": "gru.bias_hh_l0",
                # "fc.weight": "dense.weight",
                # "fc.bias": "dense.bias",
                "fc.weight": "layers.dense.weight",
                "fc.bias": "layers.dense.bias",
            }

        if toPY:
            replacement_map_rev = {v: k for k, v in replacement_map.items()}
            nps = {replacement_map_rev[k]: states[k] for k in original_keys}
        else:
            nps = {replacement_map[k]: states[k] for k in original_keys}

        if toPY:
            out_file_path = dotnet_out_path / f"{database.value}-{model.value}_{i}_py.bin"
        else:
            out_file_path = py_out_path / f"{database.value}-{model.value}_{i}_dotnet.dat"
        with open(out_file_path, "wb") as f:
            exportsd.save_state_dict(nps, f)


def transStatesFromDotNetToPy(model: run.Model, sgdLrOnly = False):
    transStatesToDotNet(model, sgdLrOnly, toPY=True)


def transStatesFromRustToPy(model: run.Model, sgdLrOnly = False):
    transStatesToRust(model, sgdLrOnly, toPY=True)


def main():
    parser = argparse.ArgumentParser(description='LeNet5 on TensorFlow')
    parser.add_argument(
        'model', type=str, choices=["lenet5", "lenet1", "vgg16", "resnet20", "lstm", "gru", "textcnn"],
        help='model must be one of ["lenet5", "lenet1", "vgg16", "resnet20", "lstm", "gru", "textcnn"]')
    parser.add_argument(
        'language', type=str, choices=["rs", "rs_py", "dotnet", "dotnet_py", "py"],
        help='one of ["rs", "rs_py", "dotnet", "dotnet_py", "py"]')
    parser.add_argument(
        '--sgd_lr_only', dest='sgd_lr_only', action='store_const',
        const=True, default=False,
        help='SGD optimizer with only learning rate (default: enable momentum and weight decay)'
    )
    args = parser.parse_args()
    print(args.model, args.language, f"sgd_lr_only? {args.sgd_lr_only}")
    model = run.Model(args.model.lower())
    trans_map = {
        "rs": transStatesToRust,
        "rs_py": transStatesFromRustToPy,
        "dotnet": transStatesToDotNet,
        "dotnet_py": transStatesFromDotNetToPy,
        "py": saveTorchScriptInTrainMode,
    }
    trans_map[args.language](model, args.sgd_lr_only)


if __name__ == '__main__':
    # showStatesInRustModel(run.Model.LSTM)
    # saveTorchScriptInTrainMode(run.Model.LeNet1, True)
    main()
