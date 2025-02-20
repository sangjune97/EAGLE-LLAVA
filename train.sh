#!/bin/bash

accelerate launch -m --mixed_precision=bf16 eagle.train.main --tmpdir /home/sangjun/EAGLE-LLAVA/eagle/ge_data/0/llava_0_150000_mufp16 --pretrainedpath ./ckpt/pretrain/state_50 --cpdir ./ckpt/finetune_pool_w_img_3e-5_150000 --configpath /home/sangjun/EAGLE-LLAVA/eagle/train/llava-1.5_7B_config.json --lr 3e-5 --bs 4 --epoch 40
