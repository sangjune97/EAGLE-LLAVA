#!/bin/bash
accelerate launch -m --mixed_precision=bf16 eagle.train.main \
 --tmpdir /data/sangjun/llava_0_259736_mufp16_llava_dataset --cpdir /data/sangjun/ckpt/token/cls/layer1/20_sharegpt_hidden \
 --configpath /home/sangjun/EAGLE-LLAVA/eagle/train/llava-1.5_7B_config.json \
 --lr 1e-4 --bs 4 --epoch 20 --data_num 259736
