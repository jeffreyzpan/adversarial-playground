#!/bin/bash

set -e 

ARCH=mnist_cnn
dataset=mnist
epochs=150
GPU=$1
INPUT_SIZE=$2

python train_models.py --dataset ${dataset} --arch ${ARCH} --gpu_ids ${GPU} --save_path ./checkpoints/${2}_${dataset}_${ARCH}_${epochs} --epochs ${epochs} --train --learning_rate 0.01 --optimizer adam --batch_size 64 --input_size ${2}

