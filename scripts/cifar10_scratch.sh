#!/bin/bash

set -e 

ARCH=$1
dataset=cifar10
epochs=200
GPU=$2

python train_models.py --dataset ${dataset} --arch ${ARCH} --gpu_ids ${GPU} --save_path ./checkpoints/${dataset}_${ARCH}_${epochs} --epochs ${epochs} --train --learning_rate 0.1 --optimizer sgd --schedule 60 120 160 --gammas 0.2 0.2 0.2 --weight_decay 5e-4


