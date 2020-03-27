#!/bin/bash

set -e 

ARCH=$1
dataset=imagenet
epochs=90
GPU=$2

python train_models.py --dataset ${dataset} --arch ${ARCH} --gpu_ids ${GPU} --save_path ./checkpoints/${dataset}_${ARCH}_${epochs} --epochs ${epochs} --train --learning_rate 0.1 --optimizer sgd --weight_decay 1e-4 --batch_size 256 --schedule 30 60 --gammas 0.1 0.1

