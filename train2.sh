#!/bin/bash

accelerate launch -m --mixed_precision=bf16 eagle.train.main --tmpdir /home/sangjun/EAGLE-LLAVA/eagle/ge_data/0/llava_0_67999_mufp16 --cpdir ./ckpt/pretrain --configpath /home/sangjun/EAGLE-LLAVA/eagle/train/llava-1.5_7B_config.json --lr 3e-5 --bs 1 --epoch 50