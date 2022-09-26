import numpy as np
import torch
from torch import FloatTensor
from torch.utils.data import Dataset


def create_dataset(data, time_len, feature_len):  # 创建以time_len个点组成的时间序列，标签序列
    data_len = len(data) - time_len
    t_data = torch.empty(data_len, time_len, feature_len)
    t_data = t_data.numpy().tolist()
    for i in range(0, data_len):
        for t in range(0, time_len):
            t_data[i][t] = data[i + t][0:feature_len]
    t_data = np.array(t_data)
    return t_data


train_data_2000 = np.load('../data/2000_train_data.npy')
train_label_2000 = np.load('../data/2000_train_data.npy')
train_data_2001 = np.load('../data/2001_train_data.npy')
train_label_2001 = np.load('../data/2001_train_data.npy')
test_data = np.load('../data/2002_test_data.npy')
test_label = np.load('../data/2002_test_data.npy')

inp_2000 = create_dataset(train_data_2000, 7, 26)
label_2000 = train_label_2000[7:, 0:24]
inp_2001 = create_dataset(train_data_2001, 7, 26)
label_2001 = train_label_2001[7:, 0:24]
train_inp = np.concatenate((inp_2000, inp_2001), axis=0)
train_label = np.concatenate((label_2000, label_2001), axis=0)
np.save('../data/train_input.npy', train_inp)
np.save('../data/train_label.npy', train_label)
inp_2002 = create_dataset(test_data, 7, 26)
label_2002 = test_label[7:, 0:24]
np.save('../data/test_input.npy', inp_2002)
np.save('../data/test_label.npy', label_2002)
