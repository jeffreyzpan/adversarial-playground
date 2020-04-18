#!/bin/bash

set -e 

ARCH=resnet18
dataset=cifar10
epochs=200
GPU=$1
INPUT_SIZE=$2

python train_models.py --dataset ${dataset} --arch ${ARCH} --gpu_ids ${GPU} --save_path ./checkpoints/${2}_${dataset}_${ARCH}_${epochs} --epochs ${epochs} --train --learning_rate 0.05 --optimizer sgd --schedule 60 120 160 --gammas 0.2 0.2 0.2 --weight_decay 1e-3 --input_size ${2}


