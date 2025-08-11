#!/bin/bash
# train_debug.sh

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false

# 训练数据列表（可按需增删）
DATA_FILES=(
    "/root/data/academic-search_10155.json"
    "/root/data/general-100k.json"
)


mkdir -p logs_debug
LOG_FILE="logs_debug/train_$(date +%F_%H%M%S).log"

nohup deepspeed --num_gpus=8 train_dft.py \
    --model_name_or_path /root/models/Qwen3-8B-Base \
    --torch_dtype bfloat16 \
    --attn_implementation flash_attention_2 \
    --trust_remote_code True \
    --data_files "${DATA_FILES[@]}" \
    --max_length 8800 \
    --preprocessing_num_workers 8 \
    --validation_split_percentage 2.0 \
    --debug_data_processing True \
    --enable_gradient_checkpointing True \
    --dft_alpha 0.7 \
    --reduce_logging True \
    --log_metrics_steps 10 \
    --bf16 True \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --learning_rate 2e-6 \
    --lr_scheduler_type cosine \
    --weight_decay 0.01 \
    --gradient_accumulation_steps 8 \
    --eval_steps 500 \
    --save_steps 500 \
    --save_total_limit 3 \
    --save_only_model 1 \
    --report_to swanlab \
    --logging_steps 10 \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 8 \
    --deepspeed ./ds_config/zero3.json \
    --output_dir output_debug \
    --logging_dir ./logs_debug \
    --save_safetensors True \
    --remove_unused_columns False \
    --ddp_find_unused_parameters False \
    --disable_tqdm False \
    > "$LOG_FILE" 2>&1 &

echo "已后台运行，PID: $!"
echo "实时查看日志: tail -f $LOG_FILE"