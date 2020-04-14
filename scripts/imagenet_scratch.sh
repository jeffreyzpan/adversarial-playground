#!/bin/bash

set -e 

ARCH=resnet18
dataset=imagenet
epochs=90
GPU=$1
INPUT_SIZE=$2
CONTRAST=$3

python train_models.py --dataset ${dataset} --arch ${ARCH} --gpu_ids ${GPU} --save_path ./checkpoints/contrast_${3}_${2}_${dataset}_${ARCH}_${epochs} --epochs ${epochs} --train --learning_rate 0.1 --optimizer sgd --weight_decay 1e-4 --batch_size 256 --schedule 30 60 --gammas 0.1 0.1 --input_size ${2} --inc_contrast ${3}

