# training-code

This repository contains code to perform supervised fine-tuning of causal language models.

Based on [HuggingFace's Trainer class](https://huggingface.co/docs/transformers/main_classes/trainer), with some extra goodies like optional xFormers and LoRA training.

## Table of contents

- [Usage](#usage)
  - [Install the required dependencies](#install-the-required-dependencies)
  - [Prepare your training data](#prepare-your-training-data)
  - [Tokenize the data](#tokenize-the-data)
  - [Start training](#start-training)
- [Other features](#other-features)
  - [LoRA](#lora)

## Usage

### Install the required dependencies

`requirements.txt` should give you an idea of what you'll need - feel free to `pip install -r requirements.txt` or install things from source depending on which versions you want.

Other packages not listed in `requirements.txt` might also be useful (e.g. `wandb`, `deepspeed` and so on).

### Prepare your training data

The training data should be a JSONL (jsonlines) line, where each line is a JSON object containing `prompt` and `generation` keys. Loss is only calculated over the tokens present in the `generation` text.

Here's an example of what a line might look like:

```json
{"prompt": "<|user|>toolformer: enabled\ntoolformer access: shell\nExecutes commands in a terminal. Input should be valid commands, and the output will be any output from running that command.\nshell(shellcommand)\nHow many lines are in the file 'file.txt'?<|model|>","generation": "There are shell('wc -l file.txt') lines in the file 'file.txt'."}
```

### Tokenize the data

With the data in hand, you should use the [tokenize_data.py](./preparation/tokenize_data.py) script to tokenize it for the model you're going to be fine-tuning. For example example:

```shell
python3 ./preparation/tokenize_data.py \
  --input-file '/data/train.jsonl' \
  --output-file '/data/train.pythia.arrow' \
  --tokenizer-path 'EleutherAI/pythia-410m-deduped' \
  --max-length 2048

python3 ./preparation/tokenize_data.py \
  --input-file '/data/eval.jsonl' \
  --output-file '/data/eval.pythia.arrow' \
  --tokenizer-path 'EleutherAI/pythia-410m-deduped' \
  --max-length 2048
```

A couple important things to note:

- This will generate fairly "bloated" files - considerably larger than the originals. Plan disk capacity accordingly.
- EOS tokens will be automatically appended at the end of `generation`, so that at inference time you can use EOS as a stopping criteria (HuggingFace's `transformers` does this by default, for example).

### Start training

The main training entrypoint is [hf_trainer.py](./training/hf_trainer.py) which, as you can probably guess, is based upon [HuggingFace's Trainer class](https://huggingface.co/docs/transformers/main_classes/trainer). As such, all of the command line arguments that can usually be passed in will work here as well.

For convenience's sake, here's a decent starting point that I use myself:

```shell
#!/usr/bin/env bash

export OMP_NUM_THREADS=4
export WANDB_PROJECT="project-name"

OUTPUT_DIR="/data/checkpoints/$WANDB_PROJECT"

MODEL_NAME='EleutherAI/pythia-410m-deduped'
TRAIN_DATASET="/data/$WANDB_PROJECT/train.pythia.arrow"
EVAL_DATASET="/data/$WANDB_PROJECT/eval.pythia.arrow"

BSZ=8

accelerate launch \
    './training/hf_trainer.py' \
    --model_name_or_path "$MODEL_NAME" \
    --train_file "$TRAIN_DATASET" \
    --eval_file "$EVAL_DATASET" \
    --output_dir "$OUTPUT_DIR" \
    --report_to "wandb" \
    --do_train --do_eval \
    --ddp_find_unused_parameters false \
    --optim 'adamw_torch_fused' \
    --seed 42 --data_seed 42 \
    --logging_first_step true --logging_steps 1 \
    --dataloader_num_workers 1 \
    --per_device_train_batch_size "$BSZ" --per_device_eval_batch_size "$BSZ" \
    --fp16 true \
    --evaluation_strategy "steps" --eval_steps 128 \
    --save_strategy "steps" --save_steps 128 \
    --gradient_accumulation_steps 8 \
    --learning_rate 1.0e-5 \
    --lr_scheduler_type "cosine" \
    --warmup_steps 64 \
    --num_train_epochs 1 \
    $@
```

## Other features

### LoRA

[hf_trainer.py](./training/hf_trainer.py) can be used to train LoRAs by using the `use_lora` argument. `lora_rank`, `lora_alpha` and `lora_dropout` can be used to configure parameters. A couple caveats though:

- It does not work when combined with FSDP. I haven't bothered fixing this because apparently FSDP + LoRA does not grant any VRAM savings. If you need optimizer/model sharding, use DeepSpeed instead for now.
- The checkpoints written out by the trainer cannot be loaded using PEFT's `from_pretrained` method. The PEFT adapter needs to be "extracted" out of the full state dict first. If training with ZeRO stage 1 or 2 (or no sharding at all), here's an example of how you might do that (untested, but hopefully the general idea is clear either way):

  ```python
  import torch
  from transformers import AutoModelForCausalLM
  from peft import LoraConfig, TaskType, get_peft_model

  BASE_MODEL = "/data/your-base-model-path-here"
  OUTPUT_DIR = "/data/your-peft-adapter-will-go-here"

  model = AutoModelForCausalLM.from_pretrained(BASE_MODEL)

  # This needs to match your training configuration _exactly_.
  peft_config = LoraConfig(
      task_type=TaskType.CAUSAL_LM,
      inference_mode=False,
      r=64,
      lora_alpha=32,
      lora_dropout=0.05,
  )
  model = get_peft_model(model, peft_config)

  full_state_dict = torch.load("/data/your-checkpoint-folder-here/pytorch_model.bin", map_location="cpu")
  model.load_state_dict(full_state_dict)

  model.save_pretrained(OUTPUT_DIR)
  ```
