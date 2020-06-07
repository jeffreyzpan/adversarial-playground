import os
import numpy as np

import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from torchvision.utils import save_image

import torchvision.datasets as datasets

from io import BytesIO
from PIL import Image

class AdjustContrast:

    def __init__(self, adjustment=1):
        self.adjustment = adjustment

    def __call__(self, x):
        return transforms.functional.adjust_contrast(x, self.adjustment)

def get_data_statistics(data_loader):
    mean = 0.
    std = 0.
    nb_samples = 0.
    for (data, labels) in data_loader:
        batch_samples = data.size(0)
        data = data.view(batch_samples, data.size(1), -1)
        mean += data.mean(2).sum(0)
        std += data.std(2).sum(0)
        nb_samples += batch_samples

    mean /= nb_samples
    std /= nb_samples
    return mean, std

def get_image_complexity(image, out, name):
    image.save(os.path.join(out, '{}.jpeg'.format(name)), format='jpeg', quality=95)
    old_size = os.stat(os.path.join(out, '{}.jpeg'.format(name))).st_size

    image_copy = Image.open(os.path.join(out, '{}.jpeg'.format(name)))
    image_copy.save(os.path.join(out, '{}_comp.jpeg'.format(name)), format='jpeg', quality=10)
    new_size = os.stat(os.path.join(out, '{}_comp.jpeg'.format(name))).st_size

    os.remove(os.path.join(out, '{}.jpeg'.format(name)))
    os.remove(os.path.join(out, '{}_comp.jpeg'.format(name)))

    return new_size / old_size
        
def generate_dataset(name, path, input_size, batch_size, num_workers, inc_contrast=1, **kwargs):

    print('==> Loading dataset {}...'.format(name))
    if name == 'imagenet':
        train_path = os.path.join(path, name, 'train')
        val_path = os.path.join(path, name, 'val')
        assert os.path.exists(train_path), train_path + ' not found'
        assert os.path.exists(val_path), val_path + ' not found'
        
        train_transform_list = [AdjustContrast(inc_contrast),
                transforms.RandomResizedCrop(input_size), 
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor()]

        test_transform_list = [AdjustContrast(inc_contrast),
                transforms.Resize(int(input_size / 0.875)),
                transforms.CenterCrop(input_size),
                transforms.ToTensor()]

        train_loader = data.DataLoader(
            datasets.ImageFolder(
                train_path, transforms.Compose(train_transform_list)),
            batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True)

        val_loader = data.DataLoader(
            datasets.ImageFolder(
                val_path, transforms.Compose(test_transform_list)),
            batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True)

        n_class = 1000

    elif name == 'cifar10':
        assert os.path.exists(path), path + ' not found'

        train_transform_list = [AdjustContrast(inc_contrast),
                transforms.RandomResizedCrop(input_size), 
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor()]

        test_transform_list = [AdjustContrast(inc_contrast),
                transforms.Resize(int(input_size / 0.875)),
                transforms.CenterCrop(input_size),
                transforms.ToTensor()]

        train_loader = data.DataLoader(
            datasets.CIFAR10(
                path, True, transforms.Compose(train_transform_list)),
            batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True)

        val_loader = data.DataLoader(
            datasets.CIFAR10(
                path, False, transforms.Compose(test_transform_list)),
            batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True)

        n_class = 10

    elif name == 'cifar100':
        assert os.path.exists(path), path + ' not found'

        train_transform_list = [AdjustContrast(inc_contrast),
                transforms.RandomResizedCrop(input_size), 
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ToTensor()]

        test_transform_list = [AdjustContrast(inc_contrast),
                transforms.Resize(int(input_size / 0.875)),
                transforms.CenterCrop(input_size),
                transforms.ToTensor()]
        
        train_loader = data.DataLoader(
            datasets.CIFAR100(
                path, True, transforms.Compose(train_transform_list)),
            batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True)

        val_loader = data.DataLoader(
            datasets.CIFAR100(
                path, False, transforms.Compose(test_transform_list)),
            batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True)

        n_class = 100

    elif name == 'mnist':
        assert os.path.exists(path), path + ' not found'

        train_transform_list = [AdjustContrast(inc_contrast),
                transforms.RandomResizedCrop(input_size), 
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor()]

        test_transform_list = [AdjustContrast(inc_contrast),
                transforms.Resize(int(input_size / 0.875)),
                transforms.CenterCrop(input_size),
                transforms.ToTensor()]

        train_loader = data.DataLoader(
            datasets.MNIST(
                path, True, transforms.Compose(train_transform_list)),
            batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True)

        val_loader = data.DataLoader(
            datasets.MNIST(
                path, False, transforms.Compose(test_transform_list)),
            batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True)

        n_class = 10

    elif name == 'fmnist':
        assert os.path.exists(path), path + ' not found'

        train_transform_list = [AdjustContrast(inc_contrast),
                transforms.RandomResizedCrop(input_size), 
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor()]

        test_transform_list = [AdjustContrast(inc_contrast),
                transforms.Resize(int(input_size / 0.875)),
                transforms.CenterCrop(input_size),
                transforms.ToTensor()]

        train_loader = data.DataLoader(
            datasets.FashionMNIST(
                path, True, transforms.Compose(train_transform_list)),
            batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True)
        val_loader = data.DataLoader(
            datasets.FashionMNIST(
                path, False, transforms.Compose(test_transform_list)),
            batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True)

        n_class = 10

    else:
        raise NotImplementedError

    return train_loader, val_loader, n_class

