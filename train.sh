#!/bin/bash
accelerate launch -m --mixed_precision=bf16 eagle.train.main_wo_img \
 --tmpdir /data/sangjun/llava_0_259736_mufp16_llava_dataset --cpdir /data/sangjun/ckpt/finetune_wo_img_1e-4_llavashare_40epoch \
 --configpath /home/sangjun/EAGLE-LLAVA/eagle/train/llava-1.5_7B_config.json \
 --lr 1e-4 --bs 4 --epoch 40
