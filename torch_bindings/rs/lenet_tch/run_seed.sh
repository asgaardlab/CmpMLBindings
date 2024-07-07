conda activate pytorch_gpu
export LIBTORCH=$LIBTORCH_GPU
export LD_LIBRARY_PATH=${LIBTORCH}/lib:$LD_LIBRARY_PATH
export PATH="${CONDA_ENV_PYTORCH_LIB}:${PATH}"
export LD_LIBRARY_PATH="${CONDA_ENV_PYTORCH_LIB}:${LD_LIBRARY_PATH}"
cargo build

# for LeNet1
#cargo run -- train lenet1 gpu 200 5
#cargo run -- deploy lenet1 gpu 0 5
#cargo run -- deploy lenet1 cpu 0 5
# sgd with learning rate only
cargo run -- train lenet1 gpu 200 5 0 true
cargo run -- deploy lenet1 gpu 0 5 0 true
cargo run -- deploy lenet1 cpu 0 5 0 true

# for LeNet5
#cargo run -- train lenet5 gpu 200 5
#cargo run -- deploy lenet5 gpu 0 5
#cargo run -- deploy lenet5 cpu 0 5
# sgd with learning rate only
cargo run -- train lenet5 gpu 200 5 0 true
cargo run -- deploy lenet5 gpu 0 5 0 true
cargo run -- deploy lenet5 cpu 0 5 0 true

# for VGG-16
#cargo run -- train vgg16 gpu 200 5
#cargo run -- deploy vgg16 gpu 0 5
#cargo run -- deploy vgg16 cpu 0 5
# sgd with learning rate only
cargo run -- train vgg16 gpu 200 5 0 true
cargo run -- deploy vgg16 gpu 0 5 0 true
cargo run -- deploy vgg16 cpu 0 5 0 true

# for LSTM
cargo run -- train lstm gpu 30 5
cargo run -- deploy lstm gpu 0 5
cargo run -- deploy lstm cpu 0 5

# for GRU
cargo run -- train gru gpu 30 5
cargo run -- deploy gru gpu 0 5
cargo run -- deploy gru cpu 0 5
