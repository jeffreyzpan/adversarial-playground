#!/bin/bash

set -e 

ARCH=$1
dataset=gtsrb
epochs=50
GPU=$2

python main.py --dataset ${dataset} --arch ${ARCH} --gpu_ids ${GPU} --resume ./checkpoints/clean_${dataset}_${ARCH}_${epochs}/model_best.pth.tar --save_path ./checkpoints/attacked_${dataset}_${ARCH}_${epochs} --evaluate

