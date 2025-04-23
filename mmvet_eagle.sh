#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

python -m model_vqa_eagle \
    --model-path llava-hf/llava-1.5-7b-hf \
    --ea-model-path /data/sangjun/ckpt/finetune_wo_img_1e-4_hidden_llava_40epoch/state_40 \
    --question-file /home/sangjun/LLaVA/playground/data/eval/mm-vet/llava-mm-vet.jsonl \
    --image-folder /home/sangjun/LLaVA/playground/data/eval/mm-vet/images \
    --answers-file /data/sangjun/ckpt/finetune_wo_img_1e-4_hidden_llava_40epoch/state_40/mmvet.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1 \
    --token-process 5 \
    --num_img_tok 0 \

#0:nothing
#1:remove
#2:pool
#3:remove_except_last
#4:remove_image_token_except_first
#5:keep_topk_image_token