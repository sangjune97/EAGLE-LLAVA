#!/bin/bash
export CUDA_VISBLE_DEVICES=0
python -m eagle.train.main --tmpdir /home/sangjun/EAGLE-LLAVA/eagle/ge_data/0/llava_0_149999_mufp16 --cpdir ./ckpt/test --configpath /home/sangjun/EAGLE-LLAVA/eagle/train/llava-1.5_7B_config.json --lr 3e-4 --bs 1 --epoch 100