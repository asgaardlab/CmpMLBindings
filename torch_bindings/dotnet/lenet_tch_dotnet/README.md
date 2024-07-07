## Selecting a Proper Version of DotNet

This project is based on [DotNet 6.0](https://dotnet.microsoft.com/en-us/download/dotnet/6.0)

```bash
export DOTNET_ROOT=$HOME/dotnet6
export PATH=$PATH:$HOME/dotnet6
```

## Run

Running the code for the experiment of training models and deploying models. 

Build the project and run:

```bash
dotnet build
# dotnet run mode device model epochs runNum
# for LeNet5
dotnet run train gpu lenet5 200 5
dotnet run deploy gpu lenet5 0 5
dotnet run deploy cpu lenet5 0 5

# for VGG16
dotnet run train gpu vgg16 200 5
dotnet run deploy gpu vgg16 0 5
dotnet run deploy cpu vgg16 0 5
```
