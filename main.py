import os, sys, shutil, time, random
import argparse

import torch
import torch.backends.cudnn as cudnn
import torchvision.datasets as dset
import torchvision.transforms as transforms
import models

import art.attacks.evasion as evasion
import art.defences as defences

model_names = sorted(name for name in models.__dict__
  if name.islower() and not name.startswith("__")
  and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='Adversarial Training Benchmarking', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--data_path', type=str, default='datasets/', help='path to dataset')
parser.add_argument('--dataset', type=str, choices=['hate', 'xrays', 'gtsrb'], help='choose dataset to benchmark adversarial training techniques on.')
parser.add_argument('--arch', metavar='ARCH', default='resnet50', help='model architecture: to evaluate robustness on (default: resnet50)')
parser.add_argument('--pretrained', type=str, default='', help='path to pretrained model')
# Hyperparameters
parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train for')
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.1, help='learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')

# Model checkpoint flags
parser.add_argument('--print_freq', default=200, type=int, metavar='N', help='print frequency (default: 200)')
parser.add_argument('--save_path', type=str, default='./', help='Folder to save checkpoints and log.')
parser.add_argument('--resume', default=None, type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')

# Experiments
parser.add_argument('--attacks', type=str, default='all', help='comma seperated string of attacks to evaluate')
parser.add_argument('--defences', type=str, default='all', help='comma seperated string of defences to evaluate')

args = parser.parse_args()

assert args.arch in model_names, 'Error: model {} not supported'.format(args.arch)
model = models.__dict__[args.arch]()
print(model)

attack_name_list = args.attacks.split(',')
attack_name_list = [i.strip().lower() for i in attack_name_list] #sanitize inputs

defence_name_list = args.defences.split(',')
defence_name_list = [i.strip().lower() for i in defence_name_list] #sanitize inputs

attack_list = {}
defence_list = {}

#initialize attacks and append to dict

if 'fgsm' in attack_name_list:
    attack_list['fgsm'] = evasion.FastGradientMethod(classifier)
if 'pgd' in attack_name_list:
    attack_list['pgd'] = evasion.ProjectedGradientDescent(classifier)
if 'hopskipjump' in attack_name_list:
    attack_list['hsj'] = evasion.HopSkipJump(classifier)
if 'query-efficient' in attack_name_list:
    raise NotImplementedError
if 'deepfool' in attack_name_list:
    attack_list['deepfool'] = evasion.DeepFool(classifier)

#initialize defenses and append to dict

if 'thermometer' in defence_name_list:
    defence_list['thermometer'] = defences.ThermometerEncoding(clip_values) #TODO figure out what actually should go here 
if 'pixeldefend' in defence_name_list:
    defence_list['pixeldefend'] = defences.PixelDefend(clip_values) #TODO figure out what goes here
if 'tvm' in defence_name_list:
    defence_list['tvm'] = defences.TotalVarMin()
if 'saddlepoint' in defence_name_list:
    defence_list['saddlepoint'] = defences.AdversarialTrainer(classifier, attacks=attack_list['pgd'])

