#!/bin/bash

set -e 

ARCH=envnetv2
dataset=urbansound8k
epochs=600
GPU=$1
FOLD=$2

python run_adv.py --dataset ${dataset} --arch ${ARCH} --gpu_ids ${GPU} --resume ./checkpoints/clean_${dataset}_${ARCH}_${epochs}_${FOLD}/model_best.pth.tar --save_path ./attack_logs/${dataset}_${ARCH}_${epochs}_${FOLD} --epochs ${epochs} --fold ${FOLD} --eval_attacks --attacks fgsm pgd deepfool --defences adv_training

