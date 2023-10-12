#!/bin/bash

# Uncomment and set the following variables correspondingly to run this script:

################## VICUNA ##################
PROMPT_VERSION=v1
# MODEL_VERSION="vicuna-v1-3-7b"
MODEL_VERSION=vicuna-v1-5-7b
################## VICUNA ##################

################## LLaMA-2 ##################
# PROMPT_VERSION="llava_llama_2"
# MODEL_VERSION="llama-2-7b-chat"
################## LLaMA-2 ##################

PRETRAIN_DATA="/home/azureuser/asr_data/ExternalData/llava-med-data/PMC-15M-pretrain-llava.json"
IMAGE_FOLDER="/home/azureuser/asr_data/ExternalData/llava-med-data/media/PMC-15M"
PRETRAIN_FOLDER="/home/azureuser/asr_data/ExternalData/models/llava-med-cmuh-$MODEL_VERSION-pretrain-5ep"
OUTPUT_FOLDER="/home/azureuser/asr_data/ExternalData/models/llava-med-cmuh-$MODEL_VERSION-finetune-1ep"

python llava/train/train_mem.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path /home/azureuser/asr_data/ExternalData/models/$MODEL_VERSION \
    --version $PROMPT_VERSION \
    --data_path $PRETRAIN_DATA \
    --image_folder $IMAGE_FOLDER \
    --vision_tower openai/clip-vit-large-patch14 \
    --pretrain_mm_mlp_adapter $PRETRAIN_FOLDER/mm_projector.bin \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir $OUTPUT_FOLDER \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 3 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb
