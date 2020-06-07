#!/bin/bash

set -e 

ARCH=resnet18
dataset=imagenet
train_epochs=90
adv_epochs=20
GPU=$1
ATTACK=$2
INPUT_SIZE=$3
CONTRAST=$4

python run_adv.py --dataset ${dataset} --arch ${ARCH} --gpu_ids ${GPU} --resume ./checkpoints/new_${dataset}_${ARCH}_${train_epochs}/model_best.pth.tar --save_path /nobackup/users/jzpan/attack_logs/bim_comp_defences_${dataset}_${ARCH}_${train_epochs} --epochs ${adv_epochs} --attacks bim --defences jpeg tvm --input_size ${3} --inc_contrast ${4}
