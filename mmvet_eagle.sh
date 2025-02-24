#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

python -m model_vqa_eagle \
    --model-path llava-hf/llava-1.5-7b-hf \
    --ea-model-path /home/sangjun/EAGLE-LLAVA/ckpt/finetune_wo_img_3e-5/state_40 \
    --question-file /home/sangjun/LLaVA/playground/data/eval/mm-vet/llava-mm-vet.jsonl \
    --image-folder /home/sangjun/LLaVA/playground/data/eval/mm-vet/images \
    --answers-file /home/sangjun/EAGLE-LLAVA/ckpt/finetune_wo_img_3e-5/state_40/test_wo_img.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

#mkdir -p ./playground/data/eval/mm-vet/results
#
#python scripts/convert_mmvet_for_eval.py \
#    --src ./playground/data/eval/mm-vet/answers/llava-v1.5-7b.jsonl \
#    --dst ./playground/data/eval/mm-vet/results/llava-v1.5-7b.json

