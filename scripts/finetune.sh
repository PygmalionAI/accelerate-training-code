#!/usr/bin/env bash

export OMP_NUM_THREADS=4
export WANDB_PROJECT="project-name"

OUTPUT_DIR="/data/checkpoints/$WANDB_PROJECT"

MODEL_NAME='meta-llama/llama-2-7b-hf'
TRAIN_DATASET="/data/$WANDB_PROJECT/train.llama.arrow"
EVAL_DATASET="/data/$WANDB_PROJECT/eval.llama.arrow"

BSZ=8

accelerate launch \
    '../training/hf_trainer.py' \
    --model_name_or_path "$MODEL_NAME" \
    --train_file "$TRAIN_DATASET" \
    --eval_file "$EVAL_DATASET" \
    --output_dir "$OUTPUT_DIR" \
    --report_to "wandb" \
    --do_train --do_eval \
    --ddp_find_unused_parameters false \
    --optim 'adamw_torch_fused' \
    --seed 0 --data_seed 0 \
    --logging_first_step true --logging_steps 1 \
    --dataloader_num_workers 1 \
    --per_device_train_batch_size "$BSZ" --per_device_eval_batch_size "$BSZ" \
    --fp16 true \
    --evaluation_strategy "steps" --eval_steps 128 \
    --save_strategy "steps" --save_steps 128 \
    --save_total_limit 2 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1.0e-5 \
    --lr_scheduler_type "cosine" \
    --warmup_steps 28 \
    --num_train_epochs 4 \
    --use_xformers \
    $@
