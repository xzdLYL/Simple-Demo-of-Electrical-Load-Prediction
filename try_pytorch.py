import torch.nn as nn
import torch

rnn = nn.GRU(
    input_size=8,
    hidden_size=8,
    num_layers=1,
    batch_first=False,
    dropout=0,
    bidirectional=False
)

t = torch.randn(50, 1000, 8)

output, hn = rnn(t)
print(output)