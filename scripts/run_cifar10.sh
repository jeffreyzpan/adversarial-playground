#!/bin/bash

set -e 

ARCH=$1
dataset=cifar10
train_epochs=350
adv_epochs=20
GPU=$2

python run_adv.py --dataset ${dataset} --arch ${ARCH} --gpu_ids ${GPU} --resume ./checkpoints/clean_${dataset}_${ARCH}_${train_epochs}/model_best.pth.tar --save_path ./checkpoints/attacked_${dataset}_${ARCH}_${train_epochs} --epochs ${adv_epochs} --attacks fgsm pgd deepfool --defences jpeg

