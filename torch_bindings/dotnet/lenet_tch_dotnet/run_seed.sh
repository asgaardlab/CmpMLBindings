curl -fSsl -O https://globalcdn.nuget.org/packages/libtorch-cuda-11.3-linux-x64-part1.1.10.0.1.nupkg

conda activate pytorch_gpu
export DOTNET_ROOT=$HOME/dotnet6
export PATH=$PATH:$HOME/dotnet6

dotnet build
# dotnet run mode device model epochs runNum
# for LeNet1
# dotnet run train gpu lenet1 200 5
# dotnet run deploy gpu lenet1 0 5
# dotnet run deploy cpu lenet1 0 5
# sgd with learning rate only
dotnet run train gpu lenet1 200 5 0 sgd_lr_only
dotnet run deploy gpu lenet1 0 5 0 sgd_lr_only
dotnet run deploy cpu lenet1 0 5 0 sgd_lr_only

# for LeNet5
# dotnet run train gpu lenet5 200 5
# dotnet run deploy gpu lenet5 0 5
# dotnet run deploy cpu lenet5 0 5
# sgd with learning rate only
dotnet run train gpu lenet5 200 5 0 sgd_lr_only
dotnet run deploy gpu lenet5 0 5 0 sgd_lr_only
dotnet run deploy cpu lenet5 0 5 0 sgd_lr_only

# for VGG16
# dotnet run train gpu vgg16 200 5
# dotnet run deploy gpu vgg16 0 5
# dotnet run deploy cpu vgg16 0 5
# sgd with learning rate only
dotnet run train gpu vgg16 200 5 0 sgd_lr_only
dotnet run deploy gpu vgg16 0 5 0 sgd_lr_only
dotnet run deploy cpu vgg16 0 5 0 sgd_lr_only

# for LSTM
dotnet run train gpu lstm 30 5
dotnet run deploy gpu lstm 0 5
dotnet run deploy cpu lstm 0 5

# for GRU
dotnet run train gpu gru 30 5
dotnet run deploy gpu gru 0 5
dotnet run deploy cpu gru 0 5
