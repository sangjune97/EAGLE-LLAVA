#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

CKPT_PATH=$1
NUM_IMG_TOK=$2

echo "CKPT_PATH: ${CKPT_PATH}"
echo "NUM_IMG_TOK: ${NUM_IMG_TOK}"

python -m model_vqa_eagle \
    --model-path /home/sangjun/.cache/huggingface/hub/models--llava-hf--llava-1.5-7b-hf/snapshots/6ceb2ed33cb8f107a781c431fe2e61574da69369 \
    --ea-model-path ${CKPT_PATH} \
    --question-file /home/sangjun/LLaVA/playground/data/eval/mm-vet/llava-mm-vet.jsonl \
    --image-folder /home/sangjun/LLaVA/playground/data/eval/mm-vet/images \
    --answers-file ${CKPT_PATH}/mmvet1.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1 \
    --token-process 5 \
    --num_img_tok ${NUM_IMG_TOK} \

#0:nothing
#1:remove
#2:pool
#3:remove_except_last
#4:remove_image_token_except_first
#5:keep_topk_image_token