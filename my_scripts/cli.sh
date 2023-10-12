#!/bin/bash
python -m llava.serve.cli \
    --model-path /home/azureuser/asr_data/ExternalData/models/llava-llama-2-13b-chat-lightning-preview \
    --image-file "/home/azureuser/12314.png" \
    --load-4bit
