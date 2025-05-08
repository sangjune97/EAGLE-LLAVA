#!/bin/bash
accelerate launch -m --mixed_precision=bf16 eagle.train.main \
 --tmpdir /data/sangjun/llava_0_50000_mufp16_llava_dataset --cpdir /data/sangjun/ckpt/token/finetune_w_img_1e-4_300_layer3 \
 --configpath /home/sangjun/EAGLE-LLAVA/eagle/train/llava-1.5_7B_layer3_config.json \
 --lr 1e-4 --bs 2 --epoch 20
