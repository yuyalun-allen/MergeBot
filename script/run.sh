#!/bin/bash

if [ -z "$1" ]; then
  echo "Please provide a param: train or eval"
  exit 1
fi

if [ "$1" == "train" ]; then
    python -m llama-recipes.recipes.finetuning.finetuning \
        --use_peft \
        --peft_method lora \
        --quantization \
        --batch_size_training 1 \
        --num_epochs 20 \
        --dataset c_n_r_dataset \
        --model_name /root/autodl-tmp/Meta-Llama-3-8B-Instruct-hf \
        --output_dir save_model \
        --save_metrics
elif [ "$1" == "eval" ]; then
    python script/evaluate_result.py \
        --model_name /root/autodl-tmp/Meta-Llama-3-8B-Instruct-hf \
        --from_peft_checkpoint save_model
else
    echo "Invalid param."
    exit 1
fi
