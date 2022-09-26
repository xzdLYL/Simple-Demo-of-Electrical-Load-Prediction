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

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


setup_seed(1)


def train(model, criterion, optimizer, loader):
    model.train()
    epoch_loss = 0
    for i, batch in enumerate(loader):
        input, label = batch
        input, label = input.transpose(1, 0).cuda(),label.cuda()
        #print(input.size())
        optimizer.zero_grad()
        prediction_result = model(input)
        label = label.squeeze()
        prediction_result = prediction_result.squeeze()

        #print(label.size())
        #print(prediction_result.size())
        loss = criterion(label, prediction_result)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(loader)


def test(model, loader):
    model = model.cuda()
    model.eval()
    out = np.zeros(shape=(0, 24))
    with no_grad():
        for i, batch in enumerate(loader):
            input, label = batch
            input, label = input.transpose(1, 0).cuda(), label.cuda()
            prediction_result = model(input)
            prediction_result = prediction_result.squeeze()
            prediction_result = prediction_result.cpu()
            # print(prediction_result.size())

            # print(prediction_result.size())
            prediction_result = np.array(prediction_result)
            out = np.concatenate((out, prediction_result))
    return out

"""
下方main函数用于训练网络，最终得到训练好的网络参数，并保存
其中定义了batch_size,epochs等网络训练参数
"""
def main(model_name=None, hidden=26):

    train_input = np.load('../data/train_input.npy')

    train_label = np.load('../data/train_label.npy')
    dataset = NumberLoader(train_input, train_label, 26)
    batch_size = 700
    epochs = 1000

    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    model = gru(hidden=hidden)

    if model_name is not None:
        model.load_state_dict(load(model_name))
    model = model.cuda()

    optimizer = optim.Adam(model.parameters())

    criterion = nn.MSELoss()
    epoch_loss = 0
    loss_save = np.zeros(shape=(epochs,1))

    for i in range(epochs):
        epoch_loss = train(model, criterion, optimizer, train_loader)
        loss_save[i] = epoch_loss
        print("epoch: {} train loss: {}".format(i, epoch_loss))



    model_name = "../trained_model_parameter/model_{0:.5f}.pt".format(epoch_loss)
    save(model.state_dict(), model_name)
    np.save('../prediction_result/train_loss.npy',loss_save)
    return model_name

"""
下方为主程序，先是制作好用于训练的dataloader，然后调用main函数完成网络训练
最后制作用于测试的dataloader，然后调用test函数进行测试，并输出预测结果，画图
"""
if __name__ == "__main__":
    hidden = 26
    nlayers = 1
    model_name = main(hidden=hidden)
    print(model_name)
    model = gru(hidden=hidden)
    # model.load_state_dict(load(model_name))
    # test(model)
    model.load_state_dict(load(model_name))
    test_input = np.load('../data/test_input.npy')
    test_label = np.load('../data/test_label.npy')
    dataset = NumberLoader(test_input, test_label, 26)
    test_loader = DataLoader(dataset, batch_size=200, shuffle=False, num_workers=0)
    a = test(model, test_loader)
    a = np.array(a)
    # test_label = test_label.transpose((1, 0))
    test_label = test_label.squeeze()
    np.save('../prediction_result/prediction_2002.npy', a)
    np.save('../prediction_result/true_value_2002.npy', test_label)

    a_flatten = a.flatten()
    test_label_flatten = test_label.flatten()
    plt.plot(a_flatten, c='red')
    plt.plot(test_label_flatten, c='blue')
    plt.show()
    plt.plot(a_flatten[3000:3024], c='red')
    plt.plot(test_label_flatten[3000:3024], c='blue')
    plt.show()
    """
    ax, sa = smooth(a, 20)
    _, label_s = smooth(test_label, 20)
    #np.save('../result/stacked_GRU1_3.npy', a)
    plt.plot(sa,linewidth=1)
    plt.plot(test_label, linewidth=1)
    plt.show()
    x = np.arange(len(test_label))
    pre_plot = plt.scatter(x, sa, s = 2, c='red')
    tru_plot = plt.scatter(x, label_s, s =1.5,c='blue')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.legend(handles=[tru_plot, pre_plot], labels=['True value', 'Prediction'], loc='upper right')
    plt.show()
    #ass1 = assessment_pre(test_label, a)
"""