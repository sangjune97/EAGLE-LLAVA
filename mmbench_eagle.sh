#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
SPLIT="mmbench_dev_20230712"

python -m model_vqa_mmbench_eagle \
    --model-path llava-hf/llava-1.5-7b-hf \
    --ea-model-path /home/sangjun/EAGLE-LLAVA/ckpt/wo_img_5e-5/state_40 \
    --question-file ./playground/data/eval/mmbench/$SPLIT.tsv \
    --answers-file /home/sangjun/EAGLE-LLAVA/ckpt/wo_img_5e-5/state_40/mmbench.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode vicuna_v1 \
    --token-process 1 \

#0:nothing
#1:remove
#2:pool
#3:remove_except_last
#4:remove_except_first
