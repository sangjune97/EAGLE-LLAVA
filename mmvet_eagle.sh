#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

python -m model_vqa_eagle \
    --model-path llava-hf/llava-1.5-7b-hf \
    --question-file /home/sangjun/LLaVA/playground/data/eval/mm-vet/llava-mm-vet.jsonl \
    --image-folder /home/sangjun/LLaVA/playground/data/eval/mm-vet/images \
    --answers-file /home/sangjun/EAGLE-LLAVA/ckpt7b_finetune_lr1e-4/state_20/test.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

#mkdir -p ./playground/data/eval/mm-vet/results
#
#python scripts/convert_mmvet_for_eval.py \
#    --src ./playground/data/eval/mm-vet/answers/llava-v1.5-7b.jsonl \
#    --dst ./playground/data/eval/mm-vet/results/llava-v1.5-7b.json

