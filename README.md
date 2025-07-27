# Unsloth-Distillation

## Install

```shell
pip install unsloth evaluate rouge_score wandb huggingface-hub
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
