#!/bin/bash

set -e 

ARCH=audio_lstm
dataset=urbansound8k
epochs=100
GPU=$1
FOLD=$2

python main.py --dataset ${dataset} --arch ${ARCH} --gpu_ids ${GPU} --save_path ./checkpoints/clean_${dataset}_${ARCH}_${epochs}_${FOLD} --epochs ${epochs} --fold ${FOLD} --train --optimizer adam --learning_rate 0.005
