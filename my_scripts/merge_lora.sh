#!/bin/bash

MODEL_VERSION=vicuna-v1-5-7b
LORA_MODEL="/home/azureuser/asr_data/ExternalData/models/llava-med-cmuh-$MODEL_VERSION-finetune-1ep"
BASE_MODEL="/home/azureuser/asr_data/ExternalData/models/$MODEL_VERSION"
MERGED_MODEL="/home/azureuser/asr_data/ExternalData/models/llava-med-cmuh-$MODEL_VERSION-merged-1ep"

python scripts/merge_lora_weights.py --model-path $LORA_MODEL --model-base $BASE_MODEL --save-model-path $MERGED_MODEL

