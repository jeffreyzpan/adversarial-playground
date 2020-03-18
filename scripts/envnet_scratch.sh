#!/bin/bash

set -e 

ARCH=$1
dataset=urbansound8k
epochs=600
GPU=$2
FOLD=$3

python train_models.py --dataset ${dataset} --arch ${ARCH} --gpu_ids ${GPU} --save_path ./checkpoints/clean_${dataset}_${ARCH}_${epochs}_${FOLD} --epochs ${epochs} --fold ${FOLD} --train --optimizer sgd --learning_rate 0.1
