#!/bin/bash
accelerate launch -m --mixed_precision=bf16 eagle.train.main \
--tmpdir /data/sangjun/llava_0_49999_mufp16 \
--cpdir /data/sangjun/ckpt/not_finetune_w_img_1e-4_13b_pool_4_only_hidden \
--configpath /home/sangjun/EAGLE-LLAVA/eagle/train/llava-1.5_13B_config.json \
--basepath /home/sangjun/.cache/huggingface/hub/models--llava-hf--llava-1.5-13b-hf/snapshots/5dda2880bda009266dda7c4baff660b95ca64540 \
--lr 1e-4 --bs 2 --epoch 20 \
