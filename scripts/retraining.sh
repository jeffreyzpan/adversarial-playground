#!/bin/bash

set -e 

ARCH=cifar_resnet18
dataset=cifar10
train_epochs=350
adv_epochs=100
GPU=$1
INPUT_SIZE=$2
CONTRAST=$3

python run_adv.py --dataset ${dataset} --arch ${ARCH} --gpu_ids ${GPU} --resume ./checkpoints/contrast_${3}_${2}_${dataset}_${ARCH}_${train_epochs}/model_best.pth.tar --save_path ./attack_logs/test_retraining_${dataset}_${ARCH}_${train_epochs} --epochs ${adv_epochs} --attacks pgd --defences adv_retraining --input_size ${2} --inc_contrast ${3} --learning_rate 0.01 --optimizer adam


