import torch
import torch.nn as nn

class LinearAdapter(nn.Module):
    def __init__(self, i_dim, o_dim):
        super().__init__()
        self.linear = nn.Linear(i_dim, o_dim, bias=False)

    def forward(self, x):
        return self.linear(x)
