#!/bin/bash

accelerate launch -m --mixed_precision=bf16 eagle.train.main --tmpdir /data/sangjun/llava_0_49999_mufp16 --cpdir /data/sangjun/ckpt/not_finetune_w_img_1e-4 --configpath /home/sangjun/EAGLE-LLAVA/eagle/train/llava-1.5_7B_config.json --lr 1e-4 --bs 2 --epoch 20
