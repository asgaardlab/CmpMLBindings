Set up the conda environment by using:

```bash
$ conda env create -f environment.yml
```

```bash
conda activate pytorch_gpu
pip install torch==1.10.2+cu113 torchvision==0.11.3+cu113 torchaudio==0.10.2+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
pip install -r requirements.txt
```