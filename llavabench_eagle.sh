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
    --question-file /home/sangjun/.cache/huggingface/hub/datasets--liuhaotian--llava-bench-in-the-wild/snapshots/9773b6fd2be88d4ced480dcd9bfd4ce76e47f79b/questions.jsonl \
    --image-folder /home/sangjun/.cache/huggingface/hub/datasets--liuhaotian--llava-bench-in-the-wild/snapshots/9773b6fd2be88d4ced480dcd9bfd4ce76e47f79b/images \
    --answers-file ${CKPT_PATH}/llavabench_t1_0.jsonl \
    --temperature 1 \
    --conv-mode vicuna_v1 \
    --token-process ${TOK_PROCESS} \
    --num_img_tok ${NUM_IMG_TOK} \

#0:nothing
#1:remove
#2:pool
#3:remove_except_last
#4:remove_image_token_except_first
#5:keep_topk_image_token