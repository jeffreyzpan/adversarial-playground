#!/bin/bash

set -e 

ARCH=cifar_resnet18
dataset=cifar10
train_epochs=350
adv_epochs=20
GPU=$1

python run_adv.py --dataset ${dataset} --arch ${ARCH} --gpu_ids ${GPU} --resume ./checkpoints/new_${dataset}_${ARCH}_${train_epochs}/model_best.pth.tar --save_path ./attack_logs/attacked_${dataset}_${ARCH}_${train_epochs} --epochs ${adv_epochs} --attacks fgsm carliniL2 pgd deepfool bim boundary
