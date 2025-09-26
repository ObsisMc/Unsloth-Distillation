# Unsloth-Distillation

## Install

### Anaconda install

```shell
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh
```

```shell
source ~/miniconda3/bin/activate
```

```shell
conda init --all
```

### Dependencies install


Need python version < 3.13

```shell
pip install unsloth evaluate rouge_score wandb huggingface-hub scikit-learn
```

For evaluation, 
```shell
pip install openai dotenv
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

### Training

```python
python trainer.py
```

Customize your parameters in these methods:
```python
# specify your dateset, model, output path and so on
# the dateset should have the same structure as https://huggingface.co/datasets/Obsismc/deepeval-synthetic-welding-QA
train(
    dataset_name=dataset_name,
    model_name=model_name,
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
    prompts=prompts,
    sft_model_root=sft_model_root,
    sft_output_dir=sft_output_dir,
    thinking=thinking,
    full_finefune=full_finefune
)
```

```python
# customize your LoRA if using PEFT method rather than full fine tuning
model = FastLanguageModel.get_peft_model(
    model,
    r=128,  # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "embed_tokens",
        "lm_head",
    ],  # Add for continual pretraining
    lora_alpha=64,
    lora_dropout=0,
    bias="none",
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing="unsloth",  # True or "unsloth" for very long context
    random_state=42,
    use_rslora=False,  # We support rank stabilized LoRA
    loftq_config=None,  # And LoftQ
)
```

```python
# customize your training config like batch size, gradient accumulation step, epoch, learning rate, saving step and so on.
trainer = UnslothTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_data,
    eval_dataset=valid_data,
    max_seq_length=max_seq_length,
    dataset_num_proc=8,
    args=UnslothTrainingArguments(
        ddp_find_unused_parameters = False,
        dataset_text_field="text",
        per_device_train_batch_size=12,
        gradient_accumulation_steps=16,
        # Use num_train_epochs and warmup_ratio for longer runs!
        # max_steps = 5,
        # warmup_steps = 10,
        warmup_ratio=0.1,
        num_train_epochs=2,
        # Select a 2 to 10x smaller learning rate for the embedding matrices!
        learning_rate=5e-5,
        embedding_learning_rate=1e-5,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.00,
        lr_scheduler_type="linear",
        seed=42,
        output_dir=sft_output_dir,
        report_to="wandb",  # Use this for WandB etc
        # run_name = "weldingbook_distill",
        save_strategy="steps",
        save_steps=19,
        # eval_steps=50,
    ),
)
```


### Evaluation
```python
python unsloth_evaluator.py
```

Customize your evaluation parameters:
```python
# Configuration
model_name = "Qwen/Qwen3-32B"  # you can evaluate model in huggingface hub
model_name = "results/model/sft/Qwen3-32B/checkpoint-31" # or evaluate your local model

extraction_model_name = "gpt-4o-mini" # openai api to extract answers in responses from evaluated model

dataset_path = 'eval_data/GPQA-V1.json'
output_root = Path("results/eval")
output_file = "qwen3_32b_lora_results_more_metric.json"

max_seq_length = 1024
load_in_4bit = False
```

The evaluation dataset should have the same structure as 
```json
[
    {
        "question": "1. 沸腾钢的特点是什么?\nA: 金属表面外层较纯.\nB: 夹杂物分布均匀.\nC: 有偏析区.\nD: 有较高的冲击韧性.\nE: Si的含水量量为1.4%.",
        "answer": [
            "A",
            "C"
        ]
    },
    {
        "question": "3. 碳钢中有益的伴生元素为:\nA: Mn.\nB: Si.\nC: Al.\nD: P.\nE: N.",
        "answer": [
            "A",
            "B",
            "C"
        ]
    }
]
```
which only contains multi-choice questions.



## Troubleshooting

> AttributeError: module 'PIL.Image' has no attribute 'Resampling'
```shell
pip install --upgrade Pillow
```


> Fail to train when `load_in_4bits=True` with multi GPUs

Currently no solution, please delete `device_map="balanced"`, which can force it to train on only one GPU. However, the training performance will be much worse (higher GPU usage and slower training speed)` than that in the machine with only 1 GPU.
