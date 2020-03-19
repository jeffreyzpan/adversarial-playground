#!/bin/bash

set -e 

ARCH=$1
dataset=mnist
train_epochs=100
adv_epochs=20
GPU=$2

python run_adv.py --dataset ${dataset} --arch ${ARCH} --gpu_ids ${GPU} --resume ./checkpoints/${dataset}_${ARCH}_${train_epochs}/model_best.pth.tar --save_path ./attack_logs/${dataset}_${ARCH}_${train_epochs} --epochs ${adv_epochs} --attacks fgsm pgd

