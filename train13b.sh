#!/bin/bash
accelerate launch -m --mixed_precision=bf16 eagle.train.main_wo_img \
--tmpdir /home/sangjun/EAGLE-LLAVA/eagle/ge_data/0/llava_0_49999_mufp16 \
--cpdir ./ckpt/not_finetune_wo_img_1e-4_13b_40epoch \
--configpath /home/sangjun/EAGLE-LLAVA/eagle/train/llava-1.5_13B_config.json \
--basepath /home/sangjun/.cache/huggingface/hub/models--llava-hf--llava-1.5-13b-hf/snapshots/5dda2880bda009266dda7c4baff660b95ca64540 \
--lr 1e-4 --bs 2 --epoch 40 \
