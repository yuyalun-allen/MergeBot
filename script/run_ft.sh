# python -m finetuning \
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