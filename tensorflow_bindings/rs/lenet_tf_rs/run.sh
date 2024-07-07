conda activate tensorflow_gpu
export CUDA_VISIBLE_DEVICES=0
export TF_FORCE_GPU_ALLOW_GROWTH=true
export PATH="${CONDA_ENV_TENSORFLOW_LIB}:${PATH}"
export LD_LIBRARY_PATH="${CONDA_ENV_TENSORFLOW_LIB}:${LD_LIBRARY_PATH}"

# For LeNet1
#cargo run --features gpu -- train lenet1 200 5
cargo run --features gpu -- deploy lenet1 0 5
cargo run -- deploy lenet1 0 5
# sgd with learning rate only
cargo run --features gpu -- deploy lenet1 0 5 true
cargo run -- deploy lenet1 0 5 true

# For LeNet5
#cargo run --features gpu -- train lenet5 200 5
cargo run --features gpu -- deploy lenet5 0 5
cargo run -- deploy lenet5 0 5
# sgd with learning rate only
cargo run --features gpu -- deploy lenet5 0 5 true
cargo run -- deploy lenet5 0 5 true

# For VGG-16
#cargo run --features gpu -- train vgg16 200 5
cargo run --features gpu -- deploy vgg16 0 5
cargo run -- deploy vgg16 0 5
# sgd with learning rate only
cargo run --features gpu -- deploy vgg16 0 5 true
cargo run -- deploy vgg16 0 5 true

# For LSTM
#cargo run --features gpu -- train lstm 30 5
cargo run --features gpu -- deploy lstm 0 5
cargo run -- deploy lstm 0 5

# For GRU
#cargo run --features gpu -- train gru 30 5
cargo run --features gpu -- deploy gru 0 5
cargo run -- deploy gru 0 5
