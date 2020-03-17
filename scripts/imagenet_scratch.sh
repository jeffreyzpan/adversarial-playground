#!/bin/bash

set -e 

ARCH=$1
dataset=imagenet
epochs=165
GPU=$2

python main.py --dataset ${dataset} --arch ${ARCH} --gpu_ids ${GPU} --save_path ./checkpoints/clean_${dataset}_${ARCH}_${epochs} --epochs ${epochs} --train --learning_rate 0.1 --optimizer sgd --weight_decay 1e-4 --batch_size 256

