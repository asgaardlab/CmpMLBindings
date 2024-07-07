## Using GPU

If you want to run this project in GPU, follow the method that is mentioned in this [issue](https://github.com/SciSharp/TensorFlow.NET/issues/813):

```bash
mkdir libtensorflow-gpu-linux-x86_64-2.6.0
cd libtensorflow-gpu-linux-x86_64-2.6.0
curl -fSsl -O https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-gpu-linux-x86_64-2.6.0.tar.gz
tar xvzf libtensorflow-gpu-linux-x86_64-2.6.0.tar.gz
export LIB_TENSORFLOW_GPU=$(pwd)  # you can add this to ~/.bashrc for future use 
cd DIR_OF_THIS_PROJECT
ln -sf $LIB_TENSORFLOW_GPU"/lib/libtensorflow.so" ./bin/Debug/net6.0/runtimes/linux-x64/native/libtensorflow.so
ln -sf $LIB_TENSORFLOW_GPU"/lib/libtensorflow_framework.so.2" ./bin/Debug/net6.0/runtimes/linux-x64/native/libtensorflow_framework.so.2
```

Also, you have to add CUDA and CuDNN path to `PATH` and `LD_LIBRARY_PATH`. For example:

```shell
export PATH="PATH_TO_CUDNN/cudnn-11.2-linux-x64-v8.1.0.77/cuda/lib64:${PATH}"
export LD_LIBRARY_PATH="PATH_TO_CUDNN/cudnn-11.2-linux-x64-v8.1.0.77/cuda/lib64:${LD_LIBRARY_PATH}"
export CUDNN_INCLUDE_DIR="PATH_TO_CUDNN/include"
export CUDNN_LIBRARY="PATH_TO_CUDNN/lib64/libcudnn.so"
```

Add the `lib` path of created conda environment to `PATH` and `LD_LIBRARY_PATH`:

```bash
$ export CONDA_ENV_TENSORFLOW_LIB='PATH_TO_ANACONDA/envs/tensorflow_gpu/lib'
$ export PATH="${CONDA_ENV_TENSORFLOW_LIB}:${PATH}"
$ export LD_LIBRARY_PATH="${CONDA_ENV_TENSORFLOW_LIB}:${LD_LIBRARY_PATH}"
```

## Selecting a Proper Version of DotNet

This project is based on [DotNet 5.0](https://dotnet.microsoft.com/en-us/download/dotnet/5.0)

```bash
$ export DOTNET_ROOT=$HOME/dotnet5
$ export PATH=$PATH:$HOME/dotnet5
```

## Build Sources

```bash
$ dotnet build
```

## Train Model

```bash
$ dotnet run train 100 5 --no-build
```
(100 epochs and run 5 times)

## Performance of training

```bash
$ dotnet run prof 100 3 --no-build
```
(100 epochs and run 3 times)

## Performance of deploying

```bash
$ dotnet run deploy --no-build
```
