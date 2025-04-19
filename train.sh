#!/bin/bash
accelerate launch -m --mixed_precision=bf16 eagle.train.main \
 --tmpdir /data/sangjun/llava_0_102024_mufp16_llavadataset --cpdir /data/sangjun/ckpt/finetune_w_img_1e-4_hidden_llava_attn_score50_20epoch \
 --configpath /home/sangjun/EAGLE-LLAVA/eagle/train/llava-1.5_7B_config.json \
 --lr 1e-4 --bs 4 --epoch 20
