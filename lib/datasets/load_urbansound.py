import os
import torch
import random
from torch.utils.data import TensorDataset
from torch.nn.utils.rnn import pad_sequence
from sklearn.preprocessing import scale
import numpy as np
import pandas as pd
import librosa

def padding(pad):
    def f(sound):
        return np.pad(sound, pad, 'constant')

    return f

def random_crop(size):
    def f(sound):
        org_size = len(sound)
        start = random.randint(0, org_size - size)
        return sound[start: start + size]

    return f

def normalize(factor):
    def f(sound):
        return sound / factor

    return f

# For testing phase
def multi_crop(input_length, n_crops):
    def f(sound):
        stride = (len(sound) - input_length) // (n_crops - 1)
        sounds = [sound[stride * i: stride * i + input_length] for i in range(n_crops)]
        return np.array(sounds)

    return f

def extract_mfcc(file_path):
    audio, sr = librosa.load(file_path)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    mfcc = scale(mfcc, axis=1)
    #transpose mfcc and return
    return mfcc.T

def generate_urbansound_dataset(data_path, test_fold_id = 1):
    meta = pd.read_csv(os.path.join(data_path, 'metadata/UrbanSound8K.csv'))
    train_feature_list = []
    train_label_list = []
    test_feature_list = []
    test_label_list = []
    for index, data in meta.iterrows():
        mfcc = extract_mfcc(os.path.join(data_path, 'audio/fold{}/{}'.format(data['fold'], data['slice_file_name'])))
        label = data['classID']
        fold = data['fold']
        if int(fold) == test_fold_id:
            test_feature_list.append(mfcc)
            test_label_list.append(label)
        else:
            train_feature_list.append(mfcc)
            train_label_list.append(label)

    train_features = np.array(train_feature_list)
    np.save('train_mfcc_features_fold{}.npy'.format(test_fold_id), train_features)
    train_labels = np.array(train_label_list)
    np.save('train_mfcc_labels_fold{}.npy'.format(test_fold_id), train_labels)
    test_features = np.array(test_feature_list)
    np.save('test_mfcc_features_fold{}.npy'.format(test_fold_id), test_features)
    test_labels = np.array(test_label_list)
    np.save('test_mfcc_labels_fold{}.npy'.format(test_fold_id), test_labels)

def load_mfcc_dataset(data_path, test_fold_id = 1):
    data_path = os.path.join(data_path, 'UrbanSound8K/mfcc')
    train_features = np.load(os.path.join(data_path, 'train_mfcc_features_fold{}.npy'.format(test_fold_id)), allow_pickle=True)
    train_labels = np.load(os.path.join(data_path, 'train_mfcc_labels_fold{}.npy'.format(test_fold_id)), allow_pickle=True).astype(np.float32) 
    test_features = np.load(os.path.join(data_path, 'test_mfcc_features_fold{}.npy'.format(test_fold_id)), allow_pickle=True)
    test_labels = np.load(os.path.join(data_path, 'test_mfcc_labels_fold{}.npy'.format(test_fold_id)), allow_pickle=True) 

    train_tensor_list = [torch.from_numpy(mfcc) for mfcc in train_features]
    cutoff_index = len(train_tensor_list)
    test_tensor_list = [torch.from_numpy(mfcc) for mfcc in test_features]
    data_list = train_tensor_list + test_tensor_list
    data_tensor = pad_sequence(data_list)
    train_tensor = data_tensor[:,:cutoff_index,:].permute(1, 2, 0)
    test_tensor = data_tensor[:,cutoff_index:,:].permute(1, 2, 0)

    return TensorDataset(train_tensor, torch.from_numpy(train_labels).long()), TensorDataset(test_tensor, torch.from_numpy(test_labels).long()) 

def load_envnet_dataset(data_root, split):
    dataset = np.load(os.path.join(data_root, 'wav{}.npz'.format(44100 // 1000)), allow_pickle=True)

    # Split to train and val
    train_sounds = []
    train_labels = []
    val_sounds = []
    val_labels = []
    for i in range(1, 11):
        sounds = dataset['fold{}'.format(i)].item()['sounds']
        labels = dataset['fold{}'.format(i)].item()['labels']
        if i == split:
            val_sounds.extend(sounds)
            val_labels.extend(labels)
        else:
            train_sounds.extend(sounds)
            train_labels.extend(labels)
    
        def train_preprocessing(example): 
            functions = [padding(66650 // 2),
                      random_crop(66650),
                      normalize(32768.0),
                      ]

            for f in functions:
                example = f(example)
            return example

        def test_preprocessing(example):
            example = librosa.util.fix_length(example, 66650)
            example = example / 32768.0
            return example

    train_sounds = np.array([train_preprocessing(i) for i in train_sounds]).reshape(len(train_sounds), 1, 1, 66650)
    val_sounds = np.array([test_preprocessing(i) for i in val_sounds]).reshape(len(val_sounds), 1, 1, 66650)
    train_labels = [int(i) for i in train_labels]
    val_labels = [int(i) for i in val_labels]
    
    # Iterator setup
    train_data = TensorDataset(torch.from_numpy(train_sounds).float(), torch.Tensor(train_labels).long())
    val_data = TensorDataset(torch.from_numpy(val_sounds).float(), torch.Tensor(val_labels).long())

    return train_data, val_data

#load_envnet_dataset('/nobackup/users/jzpan/datasets/urbansound8k/', 1)
 
'''
for i in range(1, 11):
    print('generating ' + str(i))
    generate_urbansound_dataset('/nobackup/users/jzpan/datasets/urbansound8k/', i)
'''
