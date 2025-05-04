#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export SPLIT="llava_vqav2_mscoco_test-dev2015"
EA_MODEL_PATH=$1
OUTPUT_DIR="$EA_MODEL_PATH/answers/$SPLIT"
MERGED_OUTPUT="$EA_MODEL_PATH/vqav2.jsonl"

mkdir -p "$OUTPUT_DIR"

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"
CHUNKS=${#GPULIST[@]}

echo "Using $CHUNKS GPUs: ${GPULIST[*]}"
echo "Writing chunked outputs to: $OUTPUT_DIR"

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m model_vqa_loader_eagle \
        --model-path /home/sangjun/.cache/huggingface/hub/models--llava-hf--llava-1.5-7b-hf/snapshots/6ceb2ed33cb8f107a781c431fe2e61574da69369 \
        --ea-model-path "$EA_MODEL_PATH" \
        --question-file /home/sangjun/LLaVA/playground/data/eval/vqav2/$SPLIT.jsonl \
        --image-folder /data/coco/test2015 \
        --answers-file "$OUTPUT_DIR/${CHUNKS}_${IDX}.jsonl" \
        --temperature 0 \
        --conv-mode vicuna_v1 \
        --token-process 4 \
        --num_img_tok 0 \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX &
done

wait

echo "Merging chunked outputs into: $MERGED_OUTPUT"
> "$MERGED_OUTPUT"
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat "$OUTPUT_DIR/${CHUNKS}_${IDX}.jsonl" >> "$MERGED_OUTPUT"
done

echo "Done!"