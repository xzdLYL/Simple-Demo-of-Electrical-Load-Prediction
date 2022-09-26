import torch
import numpy as np
from numpy import sign
import random
import os

def smooth(data, width):
    data_len = len(data)
    smooth_len = data_len - width + 1
    smooth_data = torch.empty(data_len)
    smooth_data = np.array(smooth_data)
    for i in range(smooth_len):
        smooth_data[i] = 0
        for j in range(width):
            smooth_data[i] += data[i + j]
        smooth_data[i] = smooth_data[i] / width
    for i in range(smooth_len, data_len, 1):
        smooth_data[i] = 0
        for j in range(data_len - i):
            smooth_data[i] += data[i + j]
        smooth_data[i] = smooth_data[i] / (data_len - i)

    x = range(data_len)
    # plt.plot(x, smooth_data)
    # plt.show()
    return x, smooth_data


def max_min_normalization(data):
    data_max = max(data)
    data_min = min(data)
    for i in range(len(data)):
        data[i] = (data[i] - data_min) / (data_max - data_min)
    return data


def general_equation(first_x, first_y, second_x, second_y):
    # 斜截式 y = kx + b
    A = second_y - first_y
    B = first_x - second_x
    C = second_x * first_y - first_x * second_y
    k = -1 * A / B
    b = -1 * C / B
    return k, b


# 输入信号序列即可(list)
def envelope_extraction(signal):
    s = signal.astype(float)
    q_u = np.zeros(s.shape)
    q_l = np.zeros(s.shape)

    # 在插值值前加上第一个值。这将强制模型对上包络和下包络模型使用相同的起点。
    # Prepend the first value of (s) to the interpolating values. This forces the model to use the same starting point for both the upper and lower envelope models.
    u_x = [0, ]  # 上包络的x序列
    u_y = [s[0], ]  # 上包络的y序列

    l_x = [0, ]  # 下包络的x序列
    l_y = [s[0], ]  # 下包络的y序列

    # 检测波峰和波谷，并分别标记它们在u_x,u_y,l_x,l_中的位置。
    # Detect peaks and troughs and mark their location in u_x,u_y,l_x,l_y respectively.

    for k in range(1, len(s) - 1):
        if (sign(s[k] - s[k - 1]) == 1) and (sign(s[k] - s[k + 1]) == 1):
            u_x.append(k)
            u_y.append(s[k])

        if (sign(s[k] - s[k - 1]) == -1) and ((sign(s[k] - s[k + 1])) == -1):
            l_x.append(k)
            l_y.append(s[k])

    u_x.append(len(s) - 1)  # 上包络与原始数据切点x
    u_y.append(s[-1])  # 对应的值

    l_x.append(len(s) - 1)  # 下包络与原始数据切点x
    l_y.append(s[-1])  # 对应的值

    # u_x,l_y是不连续的，以下代码把包络转为和输入数据相同大小的数组[便于后续处理，如滤波]
    upper_envelope_y = np.zeros(len(signal))
    lower_envelope_y = np.zeros(len(signal))

    upper_envelope_y[0] = u_y[0]  # 边界值处理
    upper_envelope_y[-1] = u_y[-1]
    lower_envelope_y[0] = l_y[0]  # 边界值处理
    lower_envelope_y[-1] = l_y[-1]

    # 上包络
    last_idx, next_idx = 0, 0
    k, b = general_equation(u_x[0], u_y[0], u_x[1], u_y[1])  # 初始的k,b
    for e in range(1, len(upper_envelope_y) - 1):

        if e not in u_x:
            v = k * e + b
            upper_envelope_y[e] = v
        else:
            idx = u_x.index(e)
            upper_envelope_y[e] = u_y[idx]
            last_idx = u_x.index(e)
            next_idx = u_x.index(e) + 1
            # 求连续两个点之间的直线方程
            k, b = general_equation(u_x[last_idx], u_y[last_idx], u_x[next_idx], u_y[next_idx])

            # 下包络
    last_idx, next_idx = 0, 0
    k, b = general_equation(l_x[0], l_y[0], l_x[1], l_y[1])  # 初始的k,b
    for e in range(1, len(lower_envelope_y) - 1):

        if e not in l_x:
            v = k * e + b
            lower_envelope_y[e] = v
        else:
            idx = l_x.index(e)
            lower_envelope_y[e] = l_y[idx]
            last_idx = l_x.index(e)
            next_idx = l_x.index(e) + 1
            # 求连续两个切点之间的直线方程
            k, b = general_equation(l_x[last_idx], l_y[last_idx], l_x[next_idx], l_y[next_idx])

            # 也可以使用三次样条进行拟合
    # u_p = interp1d(u_x,u_y, kind = 'cubic',bounds_error = False, fill_value=0.0)
    # l_p = interp1d(l_x,l_y, kind = 'cubic',bounds_error = False, fill_value=0.0)
    # for k in range(0,len(s)):
    #   q_u[k] = u_p(k)
    #   q_l[k] = l_p(k)

    return upper_envelope_y  # , lower_envelope_y


def assessment_pre(data1, data2):
    # data1是真实值
    # data2是预测值
    length = len(data1)
    rmse = 0
    mae = 0
    mape = 0
    smape = 0
    for i in range(length):
        rmse += (data1[i] - data2[i]) ** 2
        mae += abs(data1[i] - data2[i])
        mape += abs((data1[i] - data2[i]) / data2[i])
        smape += 2 * abs(data1[i] - data2[i]) / (abs(data1[i]) + abs(data2[i]))
    rmse = (rmse / length) ** 0.5
    mae = mae / length
    mape = mape / length
    smape = smape / length
    return rmse, mae, mape, smape


def seed_torch(seed=42):
    seed = int(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
