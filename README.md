# training-code

This repository contains code to perform supervised or unsupervised fine-tuning of causal language models.

Based on [HuggingFace's Trainer class](https://huggingface.co/docs/transformers/main_classes/trainer), with some extra goodies like optional xFormers and LoRA training.

## Table of contents

- [Usage](#usage)
  - [Install the required dependencies](#install-the-required-dependencies)
  - [Prepare your training data](#prepare-your-training-data)
  - [Tokenize the data](#tokenize-the-data)
  - [Start training](#start-training)
- [Other features](#other-features)
  - [LoRA](#lora)
  - [xFormers](#xformers)
  - [Unsupervised fine-tuning](#unsupervised-fine-tuning)

## Usage

### Install the required dependencies

`requirements.txt` should give you an idea of what you'll need - feel free to `pip install -r requirements.txt` or install things from source depending on which versions you want.

Other packages not listed in `requirements.txt` might also be useful (e.g. `wandb`, `deepspeed` and so on). You can run `pip install -r requirements-recommended.txt` for some of these packages (do note that xformers may be tricky to install properly at times).

### Prepare your training data

The training data should be a JSONL (jsonlines) file, where each line is a JSON object containing `prompt` and `generation` keys. Loss is only calculated over the tokens present in the `generation` text.

Here's an example of what a line might look like:

```json
{"prompt": "<|user|>toolformer: enabled\ntoolformer access: shell\nExecutes commands in a terminal. Input should be valid commands, and the output will be any output from running that command.\nshell(shellcommand)\nHow many lines are in the file 'file.txt'?<|model|>","generation": "There are shell('wc -l file.txt') lines in the file 'file.txt'."}
```

### Tokenize the data

With the data in hand, you should use the [tokenize_data_sft.py](./preparation/tokenize_data_sft.py) script to tokenize it for the model you're going to be fine-tuning. For example:

```shell
python3 ./preparation/tokenize_data_sft.py \
  --input-file '/data/train.jsonl' \
  --output-file '/data/train.pythia.arrow' \
  --tokenizer-path 'EleutherAI/pythia-410m-deduped' \
  --max-length 2048

python3 ./preparation/tokenize_data_sft.py \
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
    --save_total_limit 2 \
    --gradient_accumulation_steps 8 \
    --learning_rate 1.0e-5 \
    --lr_scheduler_type "cosine" \
    --warmup_steps 64 \
    --num_train_epochs 1 \
    $@
```

## Other features

### LoRA

[hf_trainer.py](./training/hf_trainer.py) can be used to train LoRAs by using the `use_lora` argument. `lora_rank`, `lora_alpha`, `lora_dropout` and `lora_target_modules` can be used to configure parameters. Notes:

- It does not work when combined with FSDP. I haven't bothered fixing this because apparently FSDP + LoRA does not grant any VRAM savings. If you need optimizer/model sharding, use DeepSpeed instead for now.
- In my experience, when training a LoRA on older GPUs, increasing batch size will hurt throughput so you will likely have a bunch of VRAM leftover from using `bsz=1`. If you're training on a considerable amount of data, consider using `lora_target_modules` to also include other modules. This will improve how well the model learns, at the cost of higher VRAM usage for optimizer states and lower throughput.
  - For example: by default, only `q_proj` and `v_proj` are targeted when fine-tuning LLaMA. You can include the up/down projections in the MLP (`up_proj`, `down_proj`) by using `--lora_target_modules 'up_proj,down_proj,q_proj,v_proj'`.
  - Feel free to experiment with targeting other modules as well. If using special tokens or some uncommon language, the input embeddings and the LM head are usually also worth targeting.

### xFormers

You can pass in `--use_xformers` to [hf_trainer.py](./training/hf_trainer.py) to use the `memory_efficient_attention` implementation from xFormers for GPT-J, NeoX and LLaMA-based models. **This has not been rigorously tested on anything other than LLaMA though**, so I encourage you to do a test run with/without the flag to check for strange behavior before using it on a complete training job.

### Unsupervised fine-tuning

Although this repository is meant to be used for conversational fine-tunes which is usually done with a supervised fine-tuning regime, the repo now supports *unsupervised fine-tuning* as well. However, because this repo was built with supervised fine-tuning in mind, unsupervised fine-tuning is not enabled by default; you will need to manually enable it with the `--uft` flag when running [hf_trainer.py](./training/hf_trainer.py).

For UFT, your data should be in either the form of a **singular .txt file or multiple .txt files contained within one directory**. Note that it will not skip over any .txt files in the directory, so avoid having any .txt files which you do not want the model to tokenize within the directory you provide. To tokenize unstructured data, run [tokenize_data_uft.py](./preparation/tokenize_data_uft.py) like this:

```shell
python3 ./preparation/tokenize_data_uft.py \
  --input-file '/data/train.txt' \
  --output-file '/data/train.pythia.arrow' \
  --tokenizer-path 'EleutherAI/pythia-410m-deduped' \
  --max-length 2048

python3 ./preparation/tokenize_data_uft.py \
  --input-file '/data/eval.txt' \
  --output-file '/data/eval.pythia.arrow' \
  --tokenizer-path 'EleutherAI/pythia-410m-deduped' \
  --max-length 2048
```

Please note some things:

- Much like SFT, .arrow files may take up a significantly larger amount of space on the disk than the original text files. Plan disk space accordingly.
- EOS tokens will *not* be automatically applied at the end of a generation due to the nature of unsupervised fine-tuning.