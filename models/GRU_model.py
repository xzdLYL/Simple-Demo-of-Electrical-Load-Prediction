import torch
from torch import nn
import math
import numpy as np


class gru(nn.Module):
    def __init__(self, hidden):
        super(gru, self).__init__()
        self.gru = nn.GRU(
            input_size=26,
            hidden_size=100,
            num_layers=1,
            batch_first=False,
            bidirectional=False
        )
        self.activate = torch.nn.Sigmoid()
        self.fc1 = nn.Linear(100, 24)
        self.fc2 = nn.Linear(24, 24)

    def forward(self, input):
        #print(src.size())
        #print(input.size())
        output, hn = self.gru(input)
        # print((output.size()))
        hn = np.squeeze(hn)
        #print(hn.size())
        #hn = hn[-1, :, :]
        # print(hn.size())
        # indicator = self.fc1(hn)
        indicator = self.activate(self.fc2(self.fc1(hn)))

        return indicator
