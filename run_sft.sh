export CUDA_VISIBLE_DEVICES="0"
export HF_ENDPOINT="https://hf-mirror.com"
python sft.py \
    --model_name_or_path "Qwen/Qwen-7B-Chat" \
    --bf16 True \
    --output_dir "./output_qwen" \
    --save_path "./sft_out" \
    --num_train_epochs 20 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "no" \
    --save_total_limit 1 \
    --learning_rate 3e-4 \
    --weight_decay 0.1 \
    --adam_beta2 0.95 \
    --warmup_ratio 0.01 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --report_to "tensorboard" \
    --max_length 512 \
    --gradient_checkpointing \
    --use_lora
