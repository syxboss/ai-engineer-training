# 设置环境变量
export CUDA_VISIBLE_DEVICES=0
export LOG_LEVEL=INFO

# 执行训练
swift sft \
    --model_type qwen2_5-7b-instruct \
    --model_id_or_path Qwen/Qwen2.5-7B-Instruct \
    --dataset ./data/medical_qa_data.jsonl \
    --template_type qwen \
    --sft_type lora \
    --lora_target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
    --lora_rank 16 \
    --lora_alpha 32 \
    --lora_dropout_p 0.1 \
    --num_train_epochs 3 \
    --max_length 512 \
    --batch_size 2 \
    --learning_rate 2e-5 \
    --gradient_accumulation_steps 4 \
    --warmup_ratio 0.1 \
    --weight_decay 0.01 \
    --fp16 true \
    --gradient_checkpointing true \
    --output_dir ./qwen-medical-qa-swift \
    --logging_steps 5 \
    --save_steps 100 \
    --eval_steps 100 \
    --evaluation_strategy steps \
    --save_strategy steps \
    --save_total_limit 2 \
    --load_best_model_at_end true \
    --metric_for_best_model eval_loss \
    --dataloader_num_workers 1 \
    --seed 42