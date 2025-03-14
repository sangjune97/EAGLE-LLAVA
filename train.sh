#!/bin/bash

accelerate launch -m --mixed_precision=bf16 eagle.train.main --tmpdir /home/sangjun/EAGLE-LLAVA/eagle/ge_data/0/llava_0_149999_mufp16 --cpdir ./ckpt/not_finetune_w_img_1e-4_large --configpath /home/sangjun/EAGLE-LLAVA/eagle/train/llava-1.5_7B_config.json --lr 1e-4 --bs 4 --epoch 20
