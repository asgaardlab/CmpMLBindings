# Replication package for the paper "Studying the Impact of TensorFlow and PyTorch Bindings on Machine Learning Software Quality"

This repository provides all the data and code required to reproduce our paper. 

## Jupyter Notebook to reproduce the results

We provide a notebook [analysis.ipynb](./analysis.ipynb) to reproduce the results of our paper.

## Data of model training and inference experiments

All the data can be found under the [./data](./data) folder. The folder structure is organized in
three layers (`framework -> model -> language`) as follows:

```shell
data
├── pytorch
│   ├── gru
│   │   ├── dotnet
│   │   ├── py
│   │   ├── rs
│   │   └── ts
│   ├── lenet1
│   │   └── ...
│   ├── lenet5
│   │   └── ...
│   ├── lstm
│   │   └── ...
│   └── vgg16
│   │   └── ...
└── tensorflow
    └── ...
```

each `language` (e.g., `py`) folder contains data for:

- Training accuracy in each epoch (files named `training_errors_*`)
- Test accuracy in each epoch (files named `testing_errors_*`)
- Training time cost (files named `time_cost_*`)
- Inference time and cross-binding test accuracy are stored in files name `deploy_eval_*`


## Code in each binding

All the code for building up models are shared under `tensorflow_bindings` and `torch_bindings`.

### PyTorch

Conda environment for reference: [./torch_bindings/py/environment.yml](torch_bindings/py/environment.yml).
Also, check [./torch_bindings/py/README.md](torch_bindings/py/README.md) for setting up the environment.

For each binding, please refers to the README files for more information.

### TensorFlow

Conda environment for reference: [./tensorflow_bindings/py/cv/environment.yml](tensorflow_bindings/py/cv/environment.yml).
Also, check [./tensorflow_bindings/py/cv/README.md](tensorflow_bindings/py/cv/README.md) for setting up the environment.
