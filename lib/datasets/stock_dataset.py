import os
import numpy as np
import torch
import torch.utils as utils
import pandas as pd

def generate_stocks_dataset(data_path, stock=None, window_len=10):
    os.chdir(os.path.join(data_path, 'Stocks'))
    data_list = []

    if stock==None:
        file_list = os.listdir()
    else:
        file_list = ['{}.us.txt'.format(stock)]

    for filename in file_list:
        if not filename.endswith('txt'):
            continue
        if os.path.getsize(filename) <= 0:
            continue
        df = pd.read_csv(filename, sep=',')
        label = filename.split(sep='.')[0]
        df['Label'] = filename
        df['Date'] = pd.to_datetime(df['Date'])
        data_list.append(df)

    df = data_list[0]
    split_date = list(data_list[0]["Date"][-(2*window_len+1):])[0]

    #Split the training and test set
    training_set, test_set = df[df['Date'] < split_date], df[df['Date'] >= split_date]
    training_set = training_set.drop(['Date','Label', 'OpenInt'], 1)
    test_set = test_set.drop(['Date','Label','OpenInt'], 1)

    train_list = []
    for i in range(len(training_set)-window_len):
        temp_set = training_set[i:(i+window_len)].copy()
    
        for col in list(temp_set):
            # normalize stock values for training data
            temp_set[col] = temp_set[col]/temp_set[col].iloc[0] - 1
    
        train_list.append(temp_set)

    # normalize stock values for training labels and add extra dim to match proper shape
    train_labels = torch.Tensor((training_set['Close'][window_len:].values/training_set['Close'][:-window_len].values)-1).unsqueeze(-1)
    train_list = [np.array(train_val) for train_val in train_list]
    train_tensor = torch.Tensor(train_list)

    test_list = []
    for i in range(len(test_set)-window_len):
        temp_set = test_set[i:(i+window_len)].copy()
    
        for col in list(temp_set):
            # normalize stock values for testing data
            temp_set[col] = temp_set[col]/temp_set[col].iloc[0] - 1
    
        test_list.append(temp_set)

    # normalize stock values for testing label and add extra dim to match proper shapes
    test_labels = torch.Tensor((test_set['Close'][window_len:].values/test_set['Close'][:-window_len].values)-1).unsqueeze(-1)
    test_list = [np.array(test_val) for test_val in test_list]
    test_tensor = torch.Tensor(test_list)

    return utils.data.TensorDataset(train_tensor, train_labels), utils.data.TensorDataset(test_tensor, test_labels)





