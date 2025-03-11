#!/bin/bash

accelerate launch -m --mixed_precision=bf16 eagle.train.main_wo_img --tmpdir /home/sangjun/EAGLE-LLAVA/eagle/ge_data/0/llava_0_149999_mufp16 --pretrainedpath ./ckpt/pretrain/state_50 --cpdir ./ckpt/finetune_wo_inst_5e-5_again --configpath /home/sangjun/EAGLE-LLAVA/eagle/train/llava-1.5_7B_config.json --lr 5e-5 --bs 4 --epoch 40
