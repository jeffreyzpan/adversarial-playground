#!/bin/bash

set -e 

ARCH=cifar_resnet18
dataset=cifar100
train_epochs=200
adv_epochs=20
GPU=$1

python run_adv.py --dataset ${dataset} --arch ${ARCH} --gpu_ids ${GPU} --resume ./checkpoints/new_${dataset}_${ARCH}_${train_epochs}/model_best.pth.tar --save_path ./attack_logs/jpeg_runs_${dataset}_${ARCH}_${train_epochs} --epochs ${adv_epochs} --attacks fgsm --defences jpeg

