import numpy as np
from numpy import random
from torch import save, load, no_grad
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset_construction.Dataset import NumberLoader
from models.GRU_model import gru
import torch
import matplotlib.pyplot as plt
from utils.toolbox import smooth, assessment_pre
data_2000 = np.load('../data/2000_train_data.npy')
data_2001 = np.load('../data/2001_train_data.npy')
data_2002 = np.load('../data/2002_test_data.npy')
data = np.concatenate((data_2000, data_2001),axis=0)
data = np.concatenate((data, data_2002),axis=0)
data = data[:,0:24]
data = data.flatten()

load_power, = plt.plot(data,linewidth = 0.2)
plt.xlabel('Time(hour)')
plt.ylabel('Amplitude of load')
plt.legend(handles=[load_power], labels=['power load of England (2000-2002)'], loc='upper right')
plt.show()

data_2002_1 =data_2002[0:30,0:24]
data_2002_1 = data_2002_1.flatten()
load_power_2002_1, = plt.plot(data_2002_1,linewidth = 1)
plt.xlabel('Time(hour)')
plt.ylabel('Amplitude of load')
plt.legend(handles=[load_power], labels=['power load of England (2002.1.1-2002.1.30)'], loc='upper right')
plt.show()

data_2002_1_28 =data_2002[27:29,0:24]
data_2002_1_28 = data_2002_1_28.flatten()
load_power_2002_1_28, = plt.plot(data_2002_1_28,linewidth = 1)
plt.xlabel('Time(hour)')
plt.ylabel('Amplitude of load')
plt.legend(handles=[load_power], labels=['power load of England (2002.1.28-2002.1.29)'], loc='upper right')
plt.show()


prediction_2002 = np.load('../prediction_result/prediction_2002.npy')
label_2002 = np.load('../prediction_result/true_value_2002.npy')
prediction_flatten = prediction_2002.flatten()
label_flatten = label_2002.flatten()
plt.figure(figsize=(10, 5))
pre_plot, = plt.plot(prediction_flatten[0:24*21], c='red',linewidth = 1)
lab_plot, = plt.plot(label_flatten[0:24*21],c='blue',linewidth = 1)
plt.xlabel('Time(hour)')
plt.ylabel('Amplitude of load')
plt.title('Prediction result of 2002.1.8-2002.1.28')
plt.legend(handles=[ lab_plot,pre_plot], labels=['Measured value', 'Prediction'], loc='upper right')
plt.show()

rmse = np.zeros(shape=(len(prediction_flatten)))
for i in range(len(prediction_flatten)):
    rmse[i] = (prediction_flatten[i] - label_flatten[i]) ** 2
rmse_plot,=plt.plot(rmse[0:24*21])

plt.ylim([0,0.01])
plt.xlabel('Time(hour)')
plt.ylabel('Amplitude of RMSE')
plt.title('Prediction RMSE of 2002.1.8-2002.1.28')
plt.legend(handles=[rmse_plot], labels=['RMSE'], loc='upper right')
plt.show()

loss = np.load('../prediction_result/train_loss.npy')
_, loss_smooth = smooth(loss,20)

loss_plot,=plt.plot(loss,c='blue')
loss_smooth_plot,=plt.plot(loss_smooth,c='red')
plt.xlabel('Epoch')
plt.ylabel('Loss value')
plt.title('Training loss curve')
plt.legend(handles=[loss_plot,loss_smooth_plot], labels=['loss','smoothed loss'], loc='upper right')
plt.show()

