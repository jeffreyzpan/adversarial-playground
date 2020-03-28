#!/bin/bash

set -e 

ARCH=mnist_cnn
dataset=mnist
train_epochs=100
adv_epochs=20
GPU=$1

python run_adv.py --dataset ${dataset} --arch ${ARCH} --gpu_ids ${GPU} --resume ./checkpoints/new_${dataset}_${ARCH}_${train_epochs}/model_best.pth.tar --save_path ./attack_logs/tvm_runs_${dataset}_${ARCH}_${train_epochs} --epochs ${adv_epochs} --attacks fgsm --defences thermometer --epsilons 0.05 


