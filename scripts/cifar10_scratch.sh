#!/bin/bash

set -e 

ARCH=cifar_resnet18
dataset=cifar10
epochs=350
GPU=$1
INPUT_SIZE=$2

python train_models.py --dataset ${dataset} --arch ${ARCH} --gpu_ids ${GPU} --save_path ./checkpoints/${2}_${dataset}_${ARCH}_${epochs} --epochs ${epochs} --train --learning_rate 0.1 --optimizer sgd --schedule 150 250  --gammas 0.1 0.1 --weight_decay 5e-4 --input_size ${2}


