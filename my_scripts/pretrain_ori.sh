#!/bin/bash

# To solve `GLIBCXX_3.4.29' not found issues
export LD_LIBRARY_PATH=/opt/conda/lib:$LD_LIBRARY_PATH

# Uncomment and set the following variables correspondingly to run this script:

MODEL_VERSION=vicuna-v1-5-7b
# MODEL_VERSION=llama-2-7b-chat

########### DO NOT CHANGE ###########
########### USE THIS FOR BOTH ###########
PROMPT_VERSION=plain
########### DO NOT CHANGE ###########

python llava/train/train_mem.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path /home/azureuser/asr_data/ExternalData/models/$MODEL_VERSION \
    --version $PROMPT_VERSION \
    --data_path /home/azureuser/asr_data/ExternalData/cc3m/chat_small.json \
    --image_folder /home/azureuser/asr_data/LLaVA-CC3M-Pretrain-595K/ \
    --vision_tower openai/clip-vit-large-patch14 \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir /home/azureuser/asr_data/ExternalData/models/llava-$MODEL_VERSION-pretrain-10042141 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 24000 \
    --save_total_limit 1 \
    --learning_rate 2e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 4 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb
