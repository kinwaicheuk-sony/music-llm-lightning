import torch
import torch.nn as nn

class MLPAdapter(nn.Module):
    def __init__(self, i_dim, o_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(i_dim, o_dim),
            nn.GELU(),
            nn.Linear(o_dim, o_dim),
        )
    def forward(self, x):
        return self.mlp(x)
