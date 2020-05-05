#!/bin/bash

set -e 

ARCH=mnist_cnn
dataset=fmnist
train_epochs=100
adv_epochs=20
GPU=$1

python run_adv.py --dataset ${dataset} --arch ${ARCH} --gpu_ids ${GPU} --resume ./checkpoints/contrast_1_28_${dataset}_${ARCH}_${train_epochs}/model_best.pth.tar --save_path ./attack_logs/gpu_debug_${dataset}_${ARCH}_${train_epochs} --attacks fgsm --defences jpeg 




