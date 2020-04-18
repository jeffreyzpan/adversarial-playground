#!/bin/bash

set -e 

ARCH=cifar_resnet18
dataset=cifar100
epochs=200
GPU=$1
INPUT_SIZE=$2
CONTRAST=$3

python train_models.py --dataset ${dataset} --arch ${ARCH} --gpu_ids ${GPU} --save_path ./checkpoints/contrast_${3}_${2}_${dataset}_${ARCH}_${epochs} --epochs ${epochs} --train --learning_rate 0.05 --weight_decay 0.001 --optimizer sgd --schedule 60 120 160 --gammas 0.2 0.2 0.2 --input_size ${2} --inc_contrast ${3}

