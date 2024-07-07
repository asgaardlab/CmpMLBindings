# for LeNet-1
conda activate pytorch_gpu
python run.py train gpu lenet1 200 5
python trans_models.py lenet1 rs
python trans_models.py lenet1 dotnet
python run.py deploy gpu lenet1 0 5
python run.py deploy cpu lenet1 0 5
# sgd with learning rate only
python run.py train gpu lenet1 200 5 --sgd_lr_only
python trans_models.py lenet1 rs --sgd_lr_only
python trans_models.py lenet1 dotnet --sgd_lr_only
python run.py deploy gpu lenet1 0 5 --sgd_lr_only
python run.py deploy cpu lenet1 0 5 --sgd_lr_only

# for LeNet-5
conda activate pytorch_gpu
python run.py train gpu lenet5 200 5
python trans_models.py lenet5 rs
python trans_models.py lenet5 dotnet
python run.py deploy gpu lenet5 0 5
python run.py deploy cpu lenet5 0 5
# sgd with learning rate only
python run.py train gpu lenet5 200 5 --sgd_lr_only
python trans_models.py lenet5 rs --sgd_lr_only
python trans_models.py lenet5 dotnet --sgd_lr_only
python run.py deploy gpu lenet5 0 5 --sgd_lr_only
python run.py deploy cpu lenet5 0 5 --sgd_lr_only

# for VGG-16
conda activate pytorch_gpu
python run.py train gpu vgg16 200 5
python trans_models.py vgg16 rs
python trans_models.py vgg16 dotnet
python run.py deploy gpu vgg16 0 5
python run.py deploy cpu vgg16 0 5
# sgd with learning rate only
python run.py train gpu vgg16 200 5 --sgd_lr_only
python trans_models.py vgg16 rs --sgd_lr_only
python trans_models.py vgg16 dotnet --sgd_lr_only
python run.py deploy gpu vgg16 0 5 --sgd_lr_only
python run.py deploy cpu vgg16 0 5 --sgd_lr_only

# for ResNet-20
conda activate pytorch_gpu
python run.py train gpu resnet20 200 5
python trans_models.py resnet20 rs
python trans_models.py resnet20 dotnet
python run.py deploy gpu resnet20 0 5
python run.py deploy cpu resnet20 0 5
# sgd with learning rate only
python run.py train gpu resnet20 200 5 --sgd_lr_only
python trans_models.py resnet20 rs --sgd_lr_only
python trans_models.py resnet20 dotnet --sgd_lr_only
python run.py deploy gpu resnet20 0 5 --sgd_lr_only
python run.py deploy cpu resnet20 0 5 --sgd_lr_only

# for LSTM
conda activate pytorch_gpu
python run.py train gpu lstm 30 5
python trans_models.py lstm rs
python trans_models.py lstm dotnet
python run.py deploy gpu lstm 0 5
python run.py deploy cpu lstm 0 5

# for GRU
conda activate pytorch_gpu
python run.py train gpu gru 30 5
python trans_models.py gru rs
python trans_models.py gru dotnet
python run.py deploy gpu gru 0 5
python run.py deploy cpu gru 0 5

# for TextCNN
conda activate pytorch_gpu
python run.py train gpu textcnn 30 5
#python trans_models.py textcnn rs
#python trans_models.py textcnn dotnet
python run.py deploy gpu textcnn 0 5
python run.py deploy cpu textcnn 0 5
