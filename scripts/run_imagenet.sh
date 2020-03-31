#!/bin/bash

set -e 

ARCH=resnet18
dataset=imagenet
train_epochs=90
adv_epochs=20
GPU=$1
ATTACK=$2

python run_adv.py --dataset ${dataset} --arch ${ARCH} --gpu_ids ${GPU} --resume ./checkpoints/new_${dataset}_${ARCH}_${train_epochs}/model_best.pth.tar --save_path ./attack_logs/attacked_${dataset}_${ARCH}_${train_epochs} --epochs ${adv_epochs} --attacks fgsm pgd deepfool bim 
