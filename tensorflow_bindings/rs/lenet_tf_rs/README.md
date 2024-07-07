Run the following line to build the project:

```bash
$ cargo build
```

To run this project, you have to add codes in [./src/momentum_optimizer.rs](./src/momentum_optimizer.rs) to the binding tensorflow/src/train.rs.
To run the project in CPU, change line 9 of [./Cargo.toml](./Cargo.toml) to

```
tensorflow = { version = "0.17.0" }
```

For GPU change the line to:

```
tensorflow = { version = "0.17.0", features = ["tensorflow_gpu"] }
```

Also, add CUDA and CuDNN path to `PATH` and `LD_LIBRARY_PATH`. For example:

```shell
export PATH="PATH_TO_CUDNN/cudnn-11.2-linux-x64-v8.1.0.77/cuda/lib64:PATH_TO_ANACONDA/envs/tensorflow_gpu/lib:${PATH}"
export LD_LIBRARY_PATH="PATH_TO_CUDNN/cudnn-11.2-linux-x64-v8.1.0.77/cuda/lib64:PATH_TO_ANACONDA/envs/tensorflow_gpu/lib:${LD_LIBRARY_PATH}"
```

After that, run:

```bash
$ cargo run
```