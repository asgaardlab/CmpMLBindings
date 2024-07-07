# for LeNet-1
conda activate pytorch_gpu
python run.py pred gpu lenet1 0 5 --sgd_lr_only
python run.py pred cpu lenet1 0 5 --sgd_lr_only

# for LeNet-5
python run.py pred gpu lenet5 0 5 --sgd_lr_only
python run.py pred cpu lenet5 0 5 --sgd_lr_only

# for VGG-16
python run.py pred gpu vgg16 0 5 --sgd_lr_only
python run.py pred cpu vgg16 0 5 --sgd_lr_only

# for LSTM
python run.py pred gpu lstm 0 5
python run.py pred cpu lstm 0 5

# for GRU
python run.py pred gpu gru 0 5
python run.py pred cpu gru 0 5
