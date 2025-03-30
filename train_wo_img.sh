#!/bin/bash

accelerate launch -m --mixed_precision=bf16 eagle.train.main_wo_img \
 --tmpdir /data/sangjun/llava_0_49999_mufp16_remove \
 --cpdir /data/sangjun/ckpt/not_finetune_wo_img_1e-4_sharepgt_large \
 --configpath /home/sangjun/EAGLE-LLAVA/eagle/train/llava-1.5_7B_large_config.json \
 --lr 1e-4 --bs 2 --epoch 20
