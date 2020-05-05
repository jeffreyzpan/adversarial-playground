#!/bin/bash

set -e 

ARCH=resnet18
dataset=imagenet
train_epochs=90
adv_epochs=20
GPU=$1
ATTACK=$2
EPS=$3
INPUT_SIZE=$4
CONTRAST=$5

python run_adv.py --dataset ${dataset} --arch ${ARCH} --gpu_ids ${GPU} --resume ./checkpoints/contrast_${5}_${4}_${dataset}_${ARCH}_${train_epochs}/model_best.pth.tar --save_path /nobackup/users/jzpan/attack_logs/imagenet/contrast_${5}_${4}_${2}_${3}_comp_defences_${dataset}_${ARCH}_${train_epochs} --epochs ${adv_epochs} --attacks ${ATTACK} --defences jpeg tvm --epsilons ${3} --input_size ${4} --inc_contrast ${5} --batch_size 32
