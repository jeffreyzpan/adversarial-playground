#!/bin/bash

set -e 

ARCH=$1
dataset=mnist
epochs=100
GPU=$2

python main.py --dataset ${dataset} --arch ${ARCH} --gpu_ids ${GPU} --save_path ./checkpoints/clean_${dataset}_${ARCH}_${epochs} --epochs ${epochs} --train --learning_rate 0.01 --optimizer adam --batch_size 64 

