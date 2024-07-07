#!/usr/bin/env python
# -*- coding:utf-8 -*-

import argparse

import json
from sklearn.metrics import roc_curve, auc
import run
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf
import tensorflowjs as tfjs

import h5py

def read_hdf5(path):
    weights = {}
    keys = []
    with h5py.File(path, 'r') as f: # open file
        f.visit(keys.append) # append all keys to list
        for key in keys:
            if ':' in key: # contains data if ':' in key
                # print(f[key].name)
                weights[f[key].name] = np.array(f[key])
    return weights


WEIGHTS_MAP = {
    "batch_normalization": ["gamma:0", "beta:0", "moving_mean:0", "moving_variance:0"],
    "conv2d": ["kernel:0", "bias:0"],
    "dense": ["kernel:0", "bias:0"],
}


def recursive_compare(d1, d2, level='root'):
    if isinstance(d1, dict) and isinstance(d2, dict):
        if d1.keys() != d2.keys():
            s1 = set(d1.keys())
            s2 = set(d2.keys())
            print('{:<20} + {} - {}'.format(level, s1-s2, s2-s1))
            common_keys = s1 & s2
        else:
            common_keys = set(d1.keys())

        for k in common_keys:
            recursive_compare(d1[k], d2[k], level='{}.{}'.format(level, k))

    elif isinstance(d1, list) and isinstance(d2, list):
        if len(d1) != len(d2):
            print('{:<20} len1={}; len2={}'.format(level, len(d1), len(d2)))
        common_len = min(len(d1), len(d2))

        for i in range(common_len):
            recursive_compare(d1[i], d2[i], level='{}[{}]'.format(level, i))

    else:
        if d1 != d2:
            print('{:<20} {} != {}'.format(level, d1, d2))


def match(d1, d2):
    return json.dumps(d1, sort_keys=True) == json.dumps(d2, sort_keys=True)


def weightGroupToWeights(vs: dict, layer: str):
    for layer_name, seq in WEIGHTS_MAP.items():
        if layer.startswith(layer_name):
            return [vs[layer][s] for s in seq if s in vs[layer]]
    raise ValueError("cannot covert to weights")


def calAUCROC(model_name: run.Model, lang: str, sgdLrOnly: bool):
    out_path = run.getOutputPath(model_name, lang, lrOnly=sgdLrOnly)
    run.configGPU()
    dataset_name = run.Model2DatasetName[model_name]

    _, _, x_test, y_test, batch_size = run.DatasetMap[dataset_name]()
    onehotencoder = OneHotEncoder()
    gt_labels_one_hot = onehotencoder.fit_transform(np.array(y_test).reshape(-1, 1)).toarray()
    n_classes = gt_labels_one_hot.shape[1]

    for run_num in range(5):
        print(f"running {run_num}...")
        if lang == "py":
            model_path = out_path / f"{dataset_name.value}-{model_name.value}_{run_num}"
            net = tf.keras.models.load_model(model_path)
        elif lang == "dotnet":
            net, opt, loss_func = run.ModelMap[model_name](x_test, sgdLrOnly)
            # for layer in net.layers:
            #     lw = layer.get_weights()
            #     print(f"{layer.name}: {len(lw)}: {[w.shape for w in lw]}")
            weights = read_hdf5(out_path / f"{dataset_name.value}-{model_name.value}_weights_{run_num}.h5")
            vs = {}
            for k, v in weights.items():
                _, layer, _, var = k.split("/")
                vs.setdefault(layer, {})
                vs[layer][var] = v

            for layer in net.layers:
                # print(f"setting weights for {layer.name}")
                if layer.name not in vs:
                    # print("not found in vs")
                    continue
                extracted_weights = weightGroupToWeights(vs, layer.name)
                default_weights = layer.get_weights()
                assert len(extracted_weights) == len(default_weights)
                for w1, w2 in zip(extracted_weights, default_weights):
                    assert w1.shape == w2.shape
                layer.set_weights(extracted_weights)
            net.compile(
                optimizer=opt, loss=loss_func, metrics=['accuracy'],
            )
        elif lang == "ts":
            original_net, opt, loss_func = run.ModelMap[model_name](x_test, sgdLrOnly)
            net = tfjs.converters.load_keras_model(out_path / f"model_{run_num}" / "model.json")
            # print(match(original_net.get_config(), net.get_config()))
            recursive_compare(original_net.get_config(), net.get_config())
            net.compile(
                optimizer=opt, loss=loss_func, metrics=['accuracy'],
            )
        else:
            raise ValueError("")

        # confirm the results
        _, test_acc = net.evaluate(x_test, y_test)
        testing_err_gt = np.loadtxt(out_path / f"testing_errors_{run_num}.txt")[-1]

        pred_probs = net.predict(x_test, batch_size)
        fprs = dict()
        tprs = dict()
        thresholds = dict()
        roc_aucs = dict()

        if n_classes <= 2:
            fprs[0], tprs[0], thresholds[0] = roc_curve(y_test, pred_probs[:, 0])
            roc_aucs[0] = auc(fprs[0], tprs[0])
            macro_roc_auc = roc_aucs[0]
        else:
            for i in range(n_classes):
                # fprs[i], tprs[i], thresholds[i] = roc_curve(gt_labels_one_hot[:len(pred_prob_all), i], pred_prob_all[:, i])
                fprs[i], tprs[i], thresholds[i] = roc_curve(gt_labels_one_hot[:, i], pred_probs[:, i])
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
        'model', type=str, choices=["lenet5", "lenet1", "vgg16", "resnet20", "lstm", "grurb", "gru", "textcnn"],
        help='model must be one of ["lenet5", "lenet1", "vgg16", "resnet20", "lstm", "grurb", "gru", "textcnn"]')
    parser.add_argument(
        'language', type=str, choices=["ts", "dotnet", "py"],
        help='one of ["ts", "dotnet", "py"]')
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
