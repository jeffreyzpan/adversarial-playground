#!/bin/bash

set -e 

dataset=gtsrb
epochs=100
GPU=$1

python train_pixel.py --dataset ${dataset} --gpu_ids ${GPU} --save_path ./checkpoints/${dataset}_pixel_cnn_${epochs} --epochs ${epochs}
