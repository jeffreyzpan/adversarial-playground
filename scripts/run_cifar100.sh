#!/bin/bash

set -e 

ARCH=cifar_resnet18
dataset=cifar100
train_epochs=200
adv_epochs=20
GPU=$1
INPUT_SIZE=$2

python run_adv.py --dataset ${dataset} --arch ${ARCH} --gpu_ids ${GPU} --resume ./checkpoints/${2}_${dataset}_${ARCH}_${train_epochs}/model_best.pth.tar --save_path ./attack_logs/${2}_cw_comp_defences_${dataset}_${ARCH}_${train_epochs} --epochs ${adv_epochs} --attacks carliniLinf --defences jpeg tvm --input_size ${2}

