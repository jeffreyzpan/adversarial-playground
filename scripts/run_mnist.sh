#!/bin/bash

set -e 

ARCH=mnist_cnn
dataset=mnist
train_epochs=100
adv_epochs=20
GPU=$1
INPUT_SIZE=$2
CONTRAST=$3

python run_adv.py --dataset ${dataset} --arch ${ARCH} --gpu_ids ${GPU} --resume ./checkpoints/contrast_${3}_${2}_${dataset}_${ARCH}_${train_epochs}/model_best.pth.tar --save_path ./attack_logs/contrast_${3}_${2}_cw_comp_defences_${dataset}_${ARCH}_${train_epochs} --epochs ${adv_epochs} --attacks carliniLinf --defences jpeg tvm --input_size ${2} --inc_contrast ${3} 



