This project is based on [lenet-ts](https://github.com/mortimr/lenet-ts). 

## Go for CPU or GPU

Change your directory to [./lenet_ts_cpu](./lenet_ts_cpu) for running in CPU or
[./lenet_ts_gpu](./lenet_ts_gpu) for GPU

If you are using GPU, add CUDA and CuDNN path to `PATH` and `LD_LIBRARY_PATH`. For example:

```shell
export PATH="PATH_TO_CUDNN/cudnn-11.2-linux-x64-v8.1.0.77/cuda/lib64:PATH_TO_ANACONDA/envs/tensorflow_gpu/lib:${PATH}"
export LD_LIBRARY_PATH="PATH_TO_CUDNN/cudnn-11.2-linux-x64-v8.1.0.77/cuda/lib64:PATH_TO_ANACONDA/envs/tensorflow_gpu/lib:${LD_LIBRARY_PATH}"
```

## Install dependencies

`npm install`

## Build Sources

`npm run build`

## Train Model

`npm run train 100 5`
(100 epochs and run 5 times)

## Performance of training

`npm run prof 100 3`
(100 epochs and run 3 times)

## Performance of deploying

`npm run deploy`
