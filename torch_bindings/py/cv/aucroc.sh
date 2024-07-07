conda activate pytorch_gpu

#python trans_models.py lenet1 py
#python trans_models.py lenet5 py
#python trans_models.py vgg16 py

python trans_models.py lenet1 py --sgd_lr_only
python trans_models.py lenet5 py --sgd_lr_only
python trans_models.py vgg16 py --sgd_lr_only
python trans_models.py lstm py
python trans_models.py gru py

# Python
#python cal_aucroc.py lenet1 py
#python cal_aucroc.py lenet5 py
#python cal_aucroc.py vgg16 py

python cal_aucroc.py lenet1 py --sgd_lr_only
python cal_aucroc.py lenet5 py --sgd_lr_only
python cal_aucroc.py vgg16 py --sgd_lr_only
python cal_aucroc.py lstm py
python cal_aucroc.py gru py

# C#
#python trans_models.py lenet1 dotnet_py
#python trans_models.py lenet5 dotnet_py
#python trans_models.py vgg16 dotnet_py

python trans_models.py lenet1 dotnet_py --sgd_lr_only
python trans_models.py lenet5 dotnet_py --sgd_lr_only
python trans_models.py vgg16 dotnet_py --sgd_lr_only
python trans_models.py lstm dotnet_py
python trans_models.py gru dotnet_py

python cal_aucroc.py lenet1 dotnet --sgd_lr_only
python cal_aucroc.py lenet5 dotnet --sgd_lr_only
python cal_aucroc.py vgg16 dotnet --sgd_lr_only
python cal_aucroc.py lstm dotnet
python cal_aucroc.py gru dotnet

# Rust
#python trans_models.py lenet1 rs_py
#python trans_models.py lenet5 rs_py
#python trans_models.py vgg16 rs_py

python trans_models.py lenet1 rs_py --sgd_lr_only
python trans_models.py lenet5 rs_py --sgd_lr_only
python trans_models.py vgg16 rs_py --sgd_lr_only
python trans_models.py lstm rs_py
python trans_models.py gru rs_py

python cal_aucroc.py lenet1 rs --sgd_lr_only
python cal_aucroc.py lenet5 rs --sgd_lr_only
python cal_aucroc.py vgg16 rs --sgd_lr_only
python cal_aucroc.py lstm rs
python cal_aucroc.py gru rs
