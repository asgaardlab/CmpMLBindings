# Rust binding for PyTorch

Run the following line to build the project:

```bash
$ cargo build
```

For running the code in GPU, you have to download the library from the official website: 
[https://download.pytorch.org/libtorch/cu111/libtorch-shared-with-deps-1.9.0%2Bcu111.zip](https://download.pytorch.org/libtorch/cu111/libtorch-shared-with-deps-1.9.0%2Bcu111.zip)
and add the path to `LIBTORCH` and `LD_LIBRARY_PATH`, for example: 

```bash
mkdir 
curl -fSsl -O https://download.pytorch.org/libtorch/cu113/libtorch-cxx11-abi-shared-with-deps-1.10.2%2Bcu113.zip
unzip libtorch-cxx11-abi-shared-with-deps-1.10.2+cu113.zip -d libtorch-cxx11-abi-shared-with-deps-1.10.2+cu113
```

```shell
export LIBTORCH_GPU=PATH_TO_LIBTORCH/libtorch-cxx11-abi-shared-with-deps-1.9.0+cu111/libtorch
export LIBTORCH=$LIBTORCH_GPU
export LD_LIBRARY_PATH=${LIBTORCH}/lib:$LD_LIBRARY_PATH
```

Run the project:

```shell
$ cargo run -- train vgg16 gpu 100 5
```
