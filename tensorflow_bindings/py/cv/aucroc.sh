conda activate tensorflow_gpu

# Python
#python cal_aucroc.py lenet1 py
#python cal_aucroc.py lenet5 py
#python cal_aucroc.py vgg16 py

python cal_aucroc.py lenet1 py --sgd_lr_only
python cal_aucroc.py lenet5 py --sgd_lr_only
python cal_aucroc.py vgg16 py --sgd_lr_only
python cal_aucroc.py lstm py
python cal_aucroc.py gru py
python cal_aucroc.py grurb py

# C#
python cal_aucroc.py lenet1 dotnet --sgd_lr_only
python cal_aucroc.py lenet5 dotnet --sgd_lr_only
python cal_aucroc.py vgg16 dotnet --sgd_lr_only
#python cal_aucroc.py lstm dotnet
#python cal_aucroc.py gru dotnet

# TypeScript
#python cal_aucroc.py lenet1 ts
#python cal_aucroc.py lenet5 ts
#python cal_aucroc.py vgg16 ts

python cal_aucroc.py lenet1 ts --sgd_lr_only
python cal_aucroc.py lenet5 ts --sgd_lr_only
python cal_aucroc.py vgg16 ts --sgd_lr_only
python cal_aucroc.py lstm ts
python cal_aucroc.py grurb ts
