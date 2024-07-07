Set up the conda environment by using:

```bash
$ conda env create -f environment.yml
```

Then download CuDNN 8.1.0 from [https://developer.nvidia.com/rdp/cudnn-archive](https://developer.nvidia.com/rdp/cudnn-archive) and
add CUDA and CuDNN path to `PATH` and `LD_LIBRARY_PATH`. For example:

```shell
export PATH="PATH_TO_CUDNN/cudnn-11.2-linux-x64-v8.1.0.77/cuda/lib64:PATH_TO_ANACONDA/envs/tensorflow_gpu/lib:${PATH}"
export LD_LIBRARY_PATH="PATH_TO_CUDNN/cudnn-11.2-linux-x64-v8.1.0.77/cuda/lib64:PATH_TO_ANACONDA/envs/tensorflow_gpu/lib:${LD_LIBRARY_PATH}"
```

# inspecting saved models

```bash
saved_model_cli show --dir saved_model --tag_set serve --signature_def serving_default
python /home/leo/anaconda3/envs/cuda_11_1/lib/python3.7/site-packages/tensorflow/python/tools/import_pb_to_tensorboard.py --model_dir ./saved_model --log_dir /tmp/tensorflow_logdir
```