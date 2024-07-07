conda activate tensorflow_gpu
#export PATH="${CUDNN_PATH}:${CONDA_ENV_TENSORFLOW_LIB}:${PATH}"
#export LD_LIBRARY_PATH="${CUDNN_PATH}:${CONDA_ENV_TENSORFLOW_LIB}:${LD_LIBRARY_PATH}"

# For LeNet-1
python run.py train gpu lenet1 200 5
python trans_models.py lenet1 ts
python run.py deploy gpu lenet1 0 5
python run.py deploy cpu lenet1 0 5
# sgd with learning rate only
python run.py train gpu lenet1 200 5 --sgd_lr_only
python trans_models.py lenet1 ts --sgd_lr_only
python run.py deploy gpu lenet1 0 5 --sgd_lr_only
python run.py deploy cpu lenet1 0 5 --sgd_lr_only

# For LeNet-5
python run.py train gpu lenet5 200 5
python trans_models.py lenet5 ts
python run.py deploy gpu lenet5 0 5
python run.py deploy cpu lenet5 0 5
# sgd with learning rate only
python run.py train gpu lenet5 200 5 --sgd_lr_only
python trans_models.py lenet5 ts --sgd_lr_only
python run.py deploy gpu lenet5 0 5 --sgd_lr_only
python run.py deploy cpu lenet5 0 5 --sgd_lr_only

# For VGG-16
python run.py train gpu vgg16 200 5
python trans_models.py vgg16 ts
python run.py deploy gpu vgg16 0 5
python run.py deploy cpu vgg16 0 5
# sgd with learning rate only
python run.py train gpu vgg16 200 5 --sgd_lr_only
python trans_models.py vgg16 ts --sgd_lr_only
python run.py deploy gpu vgg16 0 5 --sgd_lr_only
python run.py deploy cpu vgg16 0 5 --sgd_lr_only

# For LSTM
python run.py train gpu lstm 30 5
python trans_models.py lstm ts
python run.py deploy gpu lstm 30 5
python run.py deploy cpu lstm 30 5

# For GRU
python run.py train gpu gru 30 5
python trans_models.py gru ts
python run.py deploy gpu gru 30 5
python run.py deploy cpu gru 30 5

# For GRU (reset_after=false)
python run.py train gpu grurb 30 5
python trans_models.py grurb ts
python run.py deploy gpu grurb 30 5
python run.py deploy cpu grurb 30 5

# For TextCNN
python run.py train gpu textcnn 30 5
python trans_models.py textcnn ts
python run.py deploy gpu textcnn 30 5
python run.py deploy cpu textcnn 30 5
