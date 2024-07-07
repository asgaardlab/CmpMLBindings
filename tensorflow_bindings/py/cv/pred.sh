conda activate tensorflow_gpu
#export PATH="${CUDNN_PATH}:${CONDA_ENV_TENSORFLOW_LIB}:${PATH}"
#export LD_LIBRARY_PATH="${CUDNN_PATH}:${CONDA_ENV_TENSORFLOW_LIB}:${LD_LIBRARY_PATH}"

# For LeNet-1
python run.py pred gpu lenet1 0 5 --sgd_lr_only
python run.py pred cpu lenet1 0 5 --sgd_lr_only

# For LeNet-5
python run.py pred gpu lenet5 0 5 --sgd_lr_only
python run.py pred cpu lenet5 0 5 --sgd_lr_only

# For VGG-16
python run.py pred gpu vgg16 0 5 --sgd_lr_only
python run.py pred cpu vgg16 0 5 --sgd_lr_only

# For LSTM
python run.py pred gpu lstm 30 5
python run.py pred cpu lstm 30 5

# For GRU
python run.py pred gpu gru 30 5
python run.py pred cpu gru 30 5

# For GRU (reset_after=false)
python run.py pred gpu grurb 30 5
python run.py pred cpu grurb 30 5
