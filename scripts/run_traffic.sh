#!/bin/bash

set -e 

ARCH=$1
dataset=gtsrb
epochs=100
GPU=$2

python main.py --dataset ${dataset} --arch ${ARCH} --gpu_ids ${GPU} --resume ./checkpoints/clean_${dataset}_${ARCH}_${epochs}/model_best.pth.tar --epochs ${epochs} --eval_attacks

