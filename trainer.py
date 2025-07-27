# the whole training code

from unsloth import FastLanguageModel, FastVisionModel
import torch
import json
import numpy as np
import pandas as pd
import datasets
from datasets import Dataset
import evaluate
from evaluate import load
import math
from tqdm import tqdm
import re
import os
import random
from unsloth import FastLanguageModel
from datasets import load_dataset
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported
from unsloth import UnslothTrainer, UnslothTrainingArguments
from datasets import load_dataset

from transformers import TrainingArguments, DataCollatorForSeq2Seq
from unsloth import is_bfloat16_supported
from unsloth import UnslothTrainer, UnslothTrainingArguments
from trl import SFTTrainer
import bitsandbytes as bnb


def train(model_name, max_seq_length, dtype, load_in_4bit, dataset_name, sft_model_root, sft_output_dir, prompts):
  ## Load Model
  print("-" * 50)
  print("Loading Model...")
  model, tokenizer = FastLanguageModel.from_pretrained(
      model_name=model_name,  # Choose ANY! eg teknium/OpenHermes-2.5-Mistral-7B
      max_seq_length=max_seq_length,
      dtype=dtype,
      load_in_4bit=load_in_4bit,
      device_map = "balanced",
      # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
  )

  print("Model dtype:", next(model.parameters()).dtype)
  # for name, module in model.named_modules():
  #   if "Linear" in str(type(module)):
  #       print(f"{name}: {type(module)}")
  print(any(isinstance(m, bnb.nn.Linear4bit) for m in model.modules()))


  ## load PEFT
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
      lora_alpha=128,
      lora_dropout=0,
      bias="none",
      # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
      use_gradient_checkpointing="unsloth",  # True or "unsloth" for very long context
      random_state=42,
      use_rslora=False,  # We support rank stabilized LoRA
      loftq_config=None,  # And LoftQ
  )


  ## Load Dataset
  print("-" * 50)
  print("Loading and Preprocessing Dataset...")
  dataset = load_dataset(dataset_name, split="train")
  print(dataset[0])
  print(dataset)


  def formatting_prompts_func(qa_pairs):
      question = qa_pairs["question"]
      answer = qa_pairs["answer"]
      messages = [
          {"role": "user", "content": question},
          {"role": "assistant", "content": answer},
      ]
      text = tokenizer.apply_chat_template(
          messages, tokenize=False, add_generation_prompt=False
      )
      return {"text": text}


  dataset_text = dataset.map(formatting_prompts_func)
  print(dataset_text[0])

  dataset_split = dataset_text.train_test_split(test_size=0.2, shuffle=True, seed=42)
  train_data = dataset_split["train"]
  valid_data = dataset_split["test"]
  print(train_data)
  print(valid_data)


  ## Train
  print("-" * 50)
  print("Training Model...")

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
          max_steps = 5,
          # warmup_steps = 10,
          warmup_ratio=0.1,
          num_train_epochs=4,
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
          eval_steps=50,
      ),
  )

  trainer_stats = trainer.train(resume_from_checkpoint=False)


  ## Infer
  print("-" * 50)
  print("Model Infering...")


  def infer(prompts, tokenizer, model):
      inputs = tokenizer(
          [
              tokenizer.apply_chat_template(
                  [{"role": "user", "content": prompt}],
                  tokenize=False,
                  add_generation_prompt=True,
              )
              for prompt in prompts
          ],
          return_tensors="pt",
          padding=True,
      ).to("cuda")

      outputs = model.generate(**inputs, max_new_tokens=256, use_cache=True, repetition_penalty = 1.1)
      output_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
      for text in output_texts:
          print(text)
          print("=" * 50)

  print("Finetuned Model output:\n")
  infer(prompts, tokenizer, model)


if __name__ == "__main__":

    root_path = "results/"  # the google drive path you store data and results

    pretrain_dataset_root = os.path.join(root_path, "data/pretrain")
    sft_dataset_root = os.path.join(root_path, "data/instruct")
    
    pretrain_model_root = os.path.join(root_path, "model/pretrain")
    sft_model_root = os.path.join(root_path, "model/sft")
    
    eval_root = os.path.join(root_path, "data/eval")
    result_root = os.path.join(root_path, "result")

    max_seq_length = 1024  # Choose any! We auto support RoPE Scaling internally!
    dtype = (
        None  # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
    )
    load_in_4bit = True  # Use 4bit quantization to reduce memory usage. Can be False.
    
    ### Dataset
    dataset_name = "Obsismc/Welding-Handbook-QA"
    # dataset_name = "Obsismc/radiographic-testing-zh"
    
    prompts = [
        "Hello, how are you?",
        "Who is the president of USA?",
        "焊接领域什么是 metal matrix composites (MMCs)?",
        (
            "6. 以下共于非晶硅探测器内部信息传递和信号转换过程的描述，正确的是（  ）\n"
            "A: X 射线光子→（转换层）→ 电子→（A/D 转换器）→数字信号输出\n"
            "B: X 射线光子→（转换层）→可见光电子→〔光电二极管）→ 电子《AD 徒换器）→数字信号输出\n"
            "C: X 射线光子→〔光电二极管）→ 电子（A/D 转换器）→数字信号输出\n"
        ),
    ]
    
    # model_name = "meta-llama/Llama-3.1-8B-Instruct"
    # model_name = "Qwen/Qwen2.5-1.5B"
    # model_name = "Qwen/Qwen2.5-3B-Instruct"
    model_name = "Qwen/Qwen2.5-7B-Instruct-1M"
    model_name = "Qwen/Qwen3-32B"
    sft_output_dir = "Qwen3-32B"
    sft_output_dir = os.path.join(sft_model_root, sft_output_dir)
    
    train(
        dataset_name=dataset_name,
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
        prompts=prompts,
        sft_model_root=sft_model_root,
        sft_output_dir=sft_output_dir
    )
