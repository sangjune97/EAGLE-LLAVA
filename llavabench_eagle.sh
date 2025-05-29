#!/bin/bash
#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

CKPT_PATH=$1
TOK_PROCESS=$2
NUM_IMG_TOK=$3


echo "CKPT_PATH: ${CKPT_PATH}"
echo "TOK_PROCESS: ${TOK_PROCESS}"
echo "NUM_IMG_TOK: ${NUM_IMG_TOK}"


python -m model_vqa_eagle \
    --model-path /home/sangjun/.cache/huggingface/hub/models--llava-hf--llava-1.5-7b-hf/snapshots/6ceb2ed33cb8f107a781c431fe2e61574da69369 \
    --ea-model-path ${CKPT_PATH} \
    --question-file /home/sangjun/llava-bench-in-the-wild/questions.jsonl \
    --image-folder /home/sangjun/llava-bench-in-the-wild/images \
    --answers-file ${CKPT_PATH}/llavabench${NUM_IMG_TOK}.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1 \
    --token-process ${TOK_PROCESS} \
    --num_img_tok ${NUM_IMG_TOK} \

#0:nothing
#1:remove
#2:pool
#3:remove_except_last
#4:remove_image_token_except_first
#5:keep_topk_image_token