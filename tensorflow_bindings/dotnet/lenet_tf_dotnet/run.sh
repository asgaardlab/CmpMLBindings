conda activate tensorflow_gpu
export CUDA_VISIBLE_DEVICES=0
export TF_FORCE_GPU_ALLOW_GROWTH=true
export PATH="${CONDA_ENV_TENSORFLOW_LIB}:${PATH}"
export LD_LIBRARY_PATH="${CONDA_ENV_TENSORFLOW_LIB}:${LD_LIBRARY_PATH}"
#export PATH="${CUDNN_PATH}:${CONDA_ENV_TENSORFLOW_LIB}:${PATH}"
#export LD_LIBRARY_PATH="${CUDNN_PATH}:${CONDA_ENV_TENSORFLOW_LIB}:${LD_LIBRARY_PATH}"
export DOTNET_ROOT=$HOME/dotnet5
export PATH=$PATH:$HOME/dotnet5
dotnet build
ln -sf $LIB_TENSORFLOW_GPU"/lib/libtensorflow.so" ./bin/Debug/net5.0/runtimes/linux-x64/native/libtensorflow.so
ln -sf $LIB_TENSORFLOW_GPU"/lib/libtensorflow_framework.so.2" ./bin/Debug/net5.0/runtimes/linux-x64/native/libtensorflow_framework.so.2

# for lenet1 (sgd_lr_only)
export CUDA_VISIBLE_DEVICES=0
export TF_FORCE_GPU_ALLOW_GROWTH=true
dotnet run train gpu lenet1 200 5 --no-build
dotnet run deploy gpu lenet1 0 5 --no-build
export CUDA_VISIBLE_DEVICES=-1
dotnet run deploy cpu lenet1 0 5 --no-build

# for lenet5 (sgd_lr_only)
export CUDA_VISIBLE_DEVICES=0
export TF_FORCE_GPU_ALLOW_GROWTH=true
dotnet run train gpu lenet5 200 5 --no-build
dotnet run deploy gpu lenet5 0 5 --no-build
export CUDA_VISIBLE_DEVICES=-1
dotnet run deploy cpu lenet5 0 5 --no-build

# for vgg16 (sgd_lr_only)
export CUDA_VISIBLE_DEVICES=0
export TF_FORCE_GPU_ALLOW_GROWTH=true
dotnet run train gpu vgg16 200 5 --no-build
dotnet run deploy gpu vgg16 0 5 --no-build
export CUDA_VISIBLE_DEVICES=-1
dotnet run deploy cpu vgg16 0 5 --no-build

# for textcnn
export CUDA_VISIBLE_DEVICES=0
export TF_FORCE_GPU_ALLOW_GROWTH=true
dotnet run train gpu textcnn 30 5 --no-build
dotnet run deploy gpu textcnn 0 5 --no-build
export CUDA_VISIBLE_DEVICES=-1
dotnet run deploy cpu textcnn 0 5 --no-build

# for lstm
# dotnet run train gpu lstm 30 5 --no-build
#dotnet run deploy gpu lstm 0 5 --no-build
#dotnet run deploy cpu lstm 0 5 --no-build
