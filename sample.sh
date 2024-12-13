#!/bin/bash
CKPT="llava-v1.5-7b"
SPLIT="llava_vqav2_mscoco_test-dev2015"

python model_vqa_loader_eagle.py \
        --model-path llava-hf/llava-1.5-7b-hf \
        --question-file /home/sangjun/LLaVA/playground/data/eval/vqav2/$SPLIT.jsonl \
        --image-folder /home/sangjun/LLaVA/playground/data/eval/vqav2/test2015 \
        --answers-file ./playground/data/eval/vqav2/answers/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl \
        --temperature 0 \
        --conv-mode vicuna_v1