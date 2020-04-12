#!/bin/bash

set -e 

ARCH=mnist_cnn
dataset=fmnist
epochs=100
GPU=$1
INPUT_SIZE=$2

python train_models.py --dataset ${dataset} --arch ${ARCH} --gpu_ids ${GPU} --save_path ./checkpoints/${2}_${dataset}_${ARCH}_${epochs} --epochs ${epochs} --train --learning_rate 0.001 --optimizer adam --batch_size 64 --input_size ${2}

