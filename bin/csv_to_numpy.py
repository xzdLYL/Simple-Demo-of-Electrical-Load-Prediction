import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import torch.nn as nn
import torch
import pywt
import math

data = pd.read_csv('../data/2000.csv', header=None)
data_numpy = np.array(data)

np.save('../data/2000_train_data.npy', data_numpy)
#np.save('../data/all_train_label.npy', object_data)

data = pd.read_csv('../data/2001.csv', header=None)
data_numpy = np.array(data)

np.save('../data/2001_train_data.npy', data_numpy)

data = pd.read_csv('../data/2002.csv', header=None)
data_numpy = np.array(data)

np.save('../data/2002_test_data.npy', data_numpy)