#!/bin/bash
python -m llava.serve.cli \
    --model-path /media/volume1/aicasr/llava-llama-2-13b-chat-lightning-preview \
    --image-file "/home/t36668/tmp/MN4a54ql.jpg" \
    --load-4bit
