#!/bin/bash
python -m llava.serve.cli \
    --model-path /home/azureuser/asr_data/ExternalData/models/llava-med-cmuh-vicuna-v1-5-7b-merged-1ep \
    --image-file "/home/azureuser/12314.png" \
    --load-4bit
