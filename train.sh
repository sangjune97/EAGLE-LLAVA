#!/bin/bash
accelerate launch -m --mixed_precision=bf16 eagle.train.main \
 --tmpdir /data/sangjun/llava_0_259736_mufp16_llava_dataset --cpdir /data/sangjun/ckpt/attention_score/cls/layer2/50_embd_mustCLS_sharegpt_40epoch \
 --configpath /home/sangjun/EAGLE-LLAVA/eagle/train/llava-1.5_7B_layer2_config.json \
 --lr 1e-4 --bs 2 --epoch 40 --data_num 259736
