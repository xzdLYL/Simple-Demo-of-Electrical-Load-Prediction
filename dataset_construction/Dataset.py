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
            t_data[i][t] = data[i + t]
    t_data = np.array(t_data)
    return t_data


class NumberLoader(Dataset):
    def __init__(self, x, indicator, feature_len):
        self.x = x
        self.indicator = indicator
        self.feature_len = feature_len

    def __getitem__(self, index):
        return FloatTensor(self.x[index]), FloatTensor(self.indicator[index])

    def __len__(self):
        return len(self.x)


