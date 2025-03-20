#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

python -m model_vqa_eagle \
    --model-path llava-hf/llava-1.5-7b-hf \
    --ea-model-path /home/sangjun/EAGLE-LLAVA/ckpt/not_finetune_w_img_1e-4_large/state_11 \
    --question-file /home/sangjun/LLaVA/playground/data/eval/mm-vet/llava-mm-vet.jsonl \
    --image-folder /home/sangjun/LLaVA/playground/data/eval/mm-vet/images \
    --answers-file /home/sangjun/EAGLE-LLAVA/ckpt/not_finetune_w_img_1e-4_large/state_11/mmvet.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1 \
    --token-process 0 \

#0:nothing
#1:remove
#2:pool
#3:remove_except_last

