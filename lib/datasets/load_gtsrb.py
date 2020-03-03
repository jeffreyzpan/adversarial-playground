import os
import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms
from lib.datasets.dataset_transforms import gtsrb_transform,gtsrb_jitter_hue,gtsrb_jitter_brightness,gtsrb_jitter_saturation,gtsrb_jitter_contrast,gtsrb_rotate,gtsrb_hvflip,gtsrb_shear,gtsrb_translate,gtsrb_center,gtsrb_hflip,gtsrb_vflip

def create_gtsrb_dataset(data_path):
    # apply resizing and normalize to mean=0, std=1 (adapted from https://github.com/poojahira/gtsrb-pytorch)
    test_transform = transforms.Compose(
            [transforms.Resize((32, 32)), transforms.ToTensor()])
    # create dataset
    # augment training set with different transforms to prevent overfitting
    trainset = torch.utils.data.ConcatDataset([dset.ImageFolder(os.path.join(data_path, 'train'),
            transform=gtsrb_transform),
            dset.ImageFolder(os.path.join(data_path, 'train'),
            transform=gtsrb_jitter_brightness),dset.ImageFolder(os.path.join(data_path, 'train'),
            transform=gtsrb_jitter_hue),dset.ImageFolder(os.path.join(data_path, 'train'),
            transform=gtsrb_jitter_contrast),dset.ImageFolder(os.path.join(data_path, 'train'),
            transform=gtsrb_jitter_saturation),dset.ImageFolder(os.path.join(data_path, 'train'),
            transform=gtsrb_translate),dset.ImageFolder(os.path.join(data_path, 'train'),
            transform=gtsrb_rotate),dset.ImageFolder(os.path.join(data_path, 'train'),
            transform=gtsrb_hvflip),dset.ImageFolder(os.path.join(data_path, 'train'),
            transform=gtsrb_center),dset.ImageFolder(os.path.join(data_path, 'train'),
            transform=gtsrb_shear)]
        )
    testset = dset.ImageFolder(os.path.join(data_path, 'val'), transform=test_transform) 

    return trainset, testset
