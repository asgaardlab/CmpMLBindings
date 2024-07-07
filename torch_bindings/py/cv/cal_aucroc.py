#!/usr/bin/env python
# -*- coding:utf-8 -*-

import pathlib
import argparse

import json
import importsd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.metrics import roc_curve, auc
import torch
import run
import tch_config
import numpy as np
from sklearn.preprocessing import OneHotEncoder


def calAUCROC(model: run.Model, lang: str, sgdLrOnly: bool):
    out_path = run.getOutputPath(model, lang, lrOnly=sgdLrOnly)
    tch_config.setDeviceToGPU(0)
    dataset_name = run.Model2DatasetName[model]
    trainloader, testloader, classes = run.loadDataset(dataset_name)
    n_classes = len(classes)

    for run_num in range(5):
        print(f"running {run_num}...")
        if lang == "py":
            gpu_model_path = out_path / f"scripted_{dataset_name.value}-{model.value}_{run_num}_gpu.pt"
            if gpu_model_path.exists():
                net = torch.jit.load(gpu_model_path)
            else:
                net = torch.jit.load(out_path / f"scripted_{dataset_name.value}-{model.value}_{run_num}.pt")
        elif lang == "dotnet":
            net, _ = run.ModelMap[model](sgdLrOnly)
            with open(out_path / f'{dataset_name.value}-{model.value}_{run_num}_py.bin', 'rb') as stream:
                state_dict = importsd.load_state_dict(stream)
            net.load_state_dict(state_dict, strict=True)
        elif lang == "rs":
            net, _ = run.ModelMap[model](sgdLrOnly)
            state_dict = torch.load(out_path / f'{dataset_name.value}-{model.value}_{run_num}_py.pth')
            net.load_state_dict(state_dict, strict=True)
        else:
            raise ValueError("")
        net.to(tch_config.DEVICE)
        net.eval()

        # confirm the results
        test_acc = run.evaluation(net, testloader, run.IsBinaryEval[dataset_name])
        testing_err_gt = np.loadtxt(out_path / f"testing_errors_{run_num}.txt")[-1]

        if hasattr(testloader.dataset, "targets"):
            gt_labels = np.array(testloader.dataset.targets)
        else:
            gt_labels = testloader.dataset.tensors[1].numpy()
        onehotencoder = OneHotEncoder()
        gt_labels_one_hot = onehotencoder.fit_transform(np.array(gt_labels).reshape(-1, 1)).toarray()
        pred_prob_all = []
        for data in testloader:
            inputs, labels = data
            inputs, labels = inputs.to(tch_config.DEVICE), labels.to(tch_config.DEVICE)
            pred = net(inputs)
            if pred.shape[1] == 1:
                pred_softmax = torch.sigmoid(pred).cpu().detach().numpy()
            else:
                pred_softmax = torch.softmax(pred, 1).cpu().detach().numpy()
            pred_prob_all.append(pred_softmax)
        pred_prob_all = np.vstack(pred_prob_all)
        print(pred_prob_all.shape)
        print(gt_labels.shape)

        fprs = dict()
        tprs = dict()
        thresholds = dict()
        roc_aucs = dict()

        if n_classes <= 2:
            fprs[0], tprs[0], thresholds[0] = roc_curve(gt_labels, pred_prob_all[:, 0])
            roc_aucs[0] = auc(fprs[0], tprs[0])
            macro_roc_auc = roc_aucs[0]
        else:
            for i in range(n_classes):
                # fprs[i], tprs[i], thresholds[i] = roc_curve(gt_labels_one_hot[:len(pred_prob_all), i], pred_prob_all[:, i])
                fprs[i], tprs[i], thresholds[i] = roc_curve(gt_labels_one_hot[:, i], pred_prob_all[:, i])
                roc_aucs[i] = auc(fprs[i], tprs[i])

            # First aggregate all false positive rates
            all_fpr = np.unique(np.concatenate([fprs[i] for i in range(n_classes)]))

            # Then interpolate all ROC curves at this points
            mean_tpr = np.zeros_like(all_fpr)
            for i in range(n_classes):
                mean_tpr += np.interp(all_fpr, fprs[i], tprs[i])

            # Finally average it and compute AUC
            mean_tpr /= n_classes
            macro_roc_auc = auc(all_fpr, mean_tpr)

        auc_roc = {
            "macro_roc_auc": macro_roc_auc,
            "roc_aucs": {i: v.tolist() for i, v in roc_aucs.items()},
            "fprs": {i: v.tolist() for i, v in fprs.items()},
            "tprs": {i: v.tolist() for i, v in tprs.items()},
            "thresholds": {i: v.tolist() for i, v in thresholds.items()},
            "evaluation": [
                np.allclose(test_acc, testing_err_gt),
                test_acc, testing_err_gt
            ],
        }
        with open(out_path / f"roc_aucs_{run_num}.json", "w") as f:
            json.dump(auc_roc, f)
        print(f"running {run_num}... succeed!")


def main():
    parser = argparse.ArgumentParser(description="Calculating AUC-ROC for PyTorch's bindings")
    parser.add_argument(
        'model', type=str, choices=["lenet5", "lenet1", "vgg16", "resnet20", "lstm", "gru", "textcnn"],
        help='model must be one of ["lenet5", "lenet1", "vgg16", "resnet20", "lstm", "gru", "textcnn"]')
    parser.add_argument(
        'language', type=str, choices=["rs", "dotnet", "py"],
        help='one of ["rs", "dotnet", "py"]')
    parser.add_argument(
        '--sgd_lr_only', dest='sgd_lr_only', action='store_const',
        const=True, default=False,
        help='SGD optimizer with only learning rate (default: enable momentum and weight decay)'
    )
    args = parser.parse_args()
    print(args.model, args.language, f"sgd_lr_only? {args.sgd_lr_only}")
    model = run.Model(args.model.lower())

    calAUCROC(model, args.language, args.sgd_lr_only)

    # for i in range(len(fprs)):
    #     plt.plot(
    #         fprs[i], tprs[i], lw=2,
    #         label='ROC curve of class {0} (area = {1:0.2f})'.format(i, roc_aucs[i])
    #     )
    # plt.plot([0, 1], [0, 1], 'k--', lw=2)
    # plt.xlim([-0.05, 1.0])
    # plt.ylim([0.0, 1.05])
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('Receiver operating characteristic for multi-class data')
    # plt.legend(loc="lower right")
    # plt.show()


if __name__ == '__main__':
    main()
