import os
import torch
import torchvision.datasets as dset
from .dataset_transforms import xray_transform,xray_jitter_brightness,xray_jitter_saturation,xray_jitter_contrast,xray_hflip

def create_xrays_dataset(data_path):

    # create dataset and augment data to prevent overfitting
    trainset = torch.utils.data.ConcatDataset([dset.ImageFolder(os.path.join(data_path, 'train'),
            transform=xray_transform),
            dset.ImageFolder(os.path.join(data_path, 'train'),
            transform=xray_jitter_brightness),dset.ImageFolder(os.path.join(data_path, 'train'),
            transform=xray_jitter_contrast),dset.ImageFolder(os.path.join(data_path, 'train'),
            transform=xray_jitter_saturation),dset.ImageFolder(os.path.join(data_path, 'train'),
            transform=xray_hflip)]
        )

    testset = dset.ImageFolder(os.path.join(data_path, 'val'), transform=xray_transform) 
    return trainset, testset
