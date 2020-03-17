import os
import numpy as np

import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

def generate_dataset(name, path, input_size, batch_size, num_workers):

    print('==> Loading dataset {}...'.format(name))
    if name == 'imagenet':
        train_path = os.path.join(path, name, 'train')
        val_path = os.path.join(path, name, 'val')
        assert os.path.exists(train_path), train_path + ' not found'
        assert os.path.exists(val_path), val_path + ' not found'
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        train_loader = data.DataLoader(
            datasets.ImageFolder(
                train_path, transforms.Compose([
                    transforms.RandomResizedCrop(input_size),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ])),
            batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True)

        val_loader = data.DataLoader(
            datasets.ImageFolder(val_path, transforms.Compose([
                transforms.Resize(int(input_size / 0.875)),
                transforms.CenterCrop(input_size),
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True)

        n_class = 1000

    elif name == 'cifar10':
        assert os.path.exists(path), path + ' not found'
        normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                         std=[0.2023, 0.1994, 0.2010])
        train_loader = data.DataLoader(
            datasets.CIFAR10(
                path, True, transforms.Compose([
                    transforms.RandomResizedCrop(input_size),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ])),
            batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True)

        val_loader = data.DataLoader(
            datasets.CIFAR10(path, False, transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True)

        n_class = 10

    elif name == 'cifar100':
        assert os.path.exists(path), path + ' not found'
        normalize = transforms.Normalize(mean=[0.5070751592371323, 0.48654887331495095, 0.4409178433670343],
                                         std=[0.2673342858792401, 0.2564384629170883, 0.27615047132568404])
        train_loader = data.DataLoader(
            datasets.CIFAR100(
                path, True, transforms.Compose([
                    transforms.RandomResizedCrop(input_size),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomRotation(15),
                    transforms.ToTensor(),
                    normalize,
                ])),
            batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True)

        val_loader = data.DataLoader(
            datasets.CIFAR100(path, False, transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True)

        n_class = 100

    elif name == 'mnist':
        assert os.path.exists(path), path + ' not found'
        normalize = transforms.Normalize((0.1307,), (0.3081,))
        train_loader = data.DataLoader(
            datasets.MNIST(
                path, True, transforms.Compose([
                    transforms.RandomResizedCrop(input_size),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ])),
            batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True)

        val_loader = data.DataLoader(
            datasets.MNIST(path, False, transforms.Compose([
                transforms.Resize(int(input_size / 0.875)),
                transforms.CenterCrop(input_size),
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True)

        n_class = 10

    elif name == 'fmnist':
        assert os.path.exists(path), path + ' not found'
        normalize = transforms.Normalize((0.5,), (0.5,))
        train_loader = data.DataLoader(
            datasets.FashionMNIST(
                path, True, transforms.Compose([
                    transforms.RandomResizedCrop(input_size),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ])),
            batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True)

        val_loader = data.DataLoader(
            datasets.FashionMNIST(path, False, transforms.Compose([
                transforms.Resize(int(input_size / 0.875)),
                transforms.CenterCrop(input_size),
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True)

        n_class = 10
    else:
        raise NotImplementedError
    return train_loader, val_loader, n_class
