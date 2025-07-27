# Unsloth-Distillation

## Install

Need python version < 3.13

```shell
pip install unsloth evaluate rouge_score wandb huggingface-hub scikit-learn
```

## Config

### Huggingface Hub
You need a huggingface account to download models from huggingface
```shell
huggingface-cli login --token your_token
```

### Wandb
We use wandb for training logging
```shell
wandb login
```


## Quick Start

```python

python trainer.py
```

## Troubleshooting

> AttributeError: module 'PIL.Image' has no attribute 'Resampling'
```shell
pip install --upgrade Pillow
```


> Fail to train when `load_in_4bits=True` with multi GPUs

Currently no solution, please delete `device_map="balanced"`, which can force it to train on only one GPU. However, the training performance will be much worse (higher GPU usage and slower training speed)` than that in the machine with only 1 GPU.
