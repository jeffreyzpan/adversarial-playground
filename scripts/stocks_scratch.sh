#!/bin/bash

set -e 

ARCH=$1
dataset=stocks
epochs=5
GPU=$2

python main.py --dataset ${dataset} --arch ${ARCH} --gpu_ids ${GPU} --save_path ./checkpoints/clean_${dataset}_${ARCH}_${epochs} --epochs ${epochs} --stock goog --train --window_size 10 --learning_rate 0.001
