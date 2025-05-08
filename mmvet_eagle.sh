#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

python -m model_vqa_eagle \
    --model-path /home/sangjun/.cache/huggingface/hub/models--llava-hf--llava-1.5-7b-hf/snapshots/6ceb2ed33cb8f107a781c431fe2e61574da69369 \
    --ea-model-path /data/sangjun/ckpt/token/finetune_w_img_1e-4_300_layer2/state_20 \
    --question-file /home/sangjun/LLaVA/playground/data/eval/mm-vet/llava-mm-vet.jsonl \
    --image-folder /home/sangjun/LLaVA/playground/data/eval/mm-vet/images \
    --answers-file /data/sangjun/ckpt/token/finetune_w_img_1e-4_300_layer2/state_20/mmvet.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1 \
    --token-process 5 \
    --num_img_tok 300 \

#0:nothing
#1:remove
#2:pool
#3:remove_except_last
#4:remove_image_token_except_first
#5:keep_topk_image_token