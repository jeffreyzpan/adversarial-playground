import os
import pandas as pd
import subprocess

base_path = '/nobackup/users/jzpan/datasets/gtsrb'

val_meta = pd.read_csv(os.path.join(base_path, 'Test.csv'))
train_meta = pd.read_csv(os.path.join(base_path, 'Train.csv'))

val_classes = val_meta['ClassId'].values
train_classes = train_meta['ClassId'].values

val_paths = val_meta['Path'].values
train_paths = train_meta['Path'].values

val_info = zip(val_classes, val_paths)
train_info = zip(train_classes, train_paths)

for i in range(43):
    subprocess.call('mkdir {}/val/{}'.format(base_path, i), shell=True)
#    subprocess.call('mkdir {}/train/{}'.format(base_path, i), shell=True)

for info in val_info:
    #try:
    subprocess.call('mv {}/val/{} {}/val/{}/'.format(base_path, info[1].split('/')[1], base_path, info[0]), shell=True)
    #subprocess.call('ls {}/val/{}'.format(base_path, info[1].split('/')[1]))
    #except:
    #    continue
