#!/bin/bash

set -e 

ARCH=$1
dataset=cifar10
epochs=350
GPU=$2

python main.py --dataset ${dataset} --arch ${ARCH} --gpu_ids ${GPU} --save_path ./checkpoints/clean_${dataset}_${ARCH}_${epochs} --epochs ${epochs} --train --learning_rate 0.1 --optimizer sgd --schedule 150 250 --gammas 0.1 0.1 --weight_decay 5e-4

