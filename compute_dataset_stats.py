import os
import sys
import argparse

import torch
import torch.utils.data as data
import torchvision.datasets as datasets
import numpy as np
import scipy
from scipy import ndimage
from lib.datasets.data_utils import get_image_complexity, generate_dataset
import multiprocessing as mp
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Calculate dataset complexity',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--data_path', type=str,
                    default='/nobackup/users/jzpan/datasets', help='path to dataset')
parser.add_argument('--dataset', type=str,
                    help='choose dataset to compute complexity.')
parser.add_argument('--temp', type=str, default='/nobackup/users/jzpan/tmp', help='path to store temp compressed images')

if __name__ == '__main__':
    args =  parser.parse_args()
    path = args.data_path
    # set variables based on dataset to evaluate on
    if args.dataset == 'imagenet':
        train_set = datasets.ImageFolder(os.path.join(path, 'imagenet/train'))
        val_set = datasets.ImageFolder(os.path.join(path, 'imagenet/val'))

    elif args.dataset == 'cifar10':
        train_set = datasets.CIFAR10(path, True)
        val_set = datasets.CIFAR10(path, False) 

    elif args.dataset == 'cifar100':
        train_set = datasets.CIFAR100(path, True)
        val_set = datasets.CIFAR100(path, False)

    elif args.dataset == 'fmnist':
        train_set = datasets.FashionMNIST(path, True)
        val_set = datasets.FashionMNIST(path, False)

    elif args.dataset  == 'mnist':
        train_set = datasets.MNIST(path, True)
        val_set = datasets.MNIST(path, False)

    pool = mp.Pool(processes=20)
    results = [pool.apply_async(get_image_complexity, args=(image[0], args.temp, str(i))) for i, image in tqdm(enumerate(train_set))]
    output = [i.get() for i in results]

    train_complexity = np.array(output)

    results = [pool.apply_async(get_image_complexity, args=(image[0], args.temp, str(i))) for i, image in tqdm(enumerate(val_set))]
    output = [i.get() for i in results]

    test_complexity = np.array(output)

    train_mean = np.mean(train_complexity)
    train_std = np.std(train_complexity)

    test_mean = np.mean(test_complexity)
    test_std = np.std(test_complexity)

    print(train_mean)
    print(train_std)
    print(test_mean)
    print(test_std)



