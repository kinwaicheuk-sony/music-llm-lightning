"""
MLP adapter for projecting audio features to LLM space.
"""
import torch
import torch.nn as nn

class MLPAdapter(nn.Module):
    """Two-layer MLP adapter with GELU activation."""

    def __init__(self, i_dim: int, o_dim: int):
        """
        Initialize MLP adapter.

        Args:
            i_dim: Input dimension
            o_dim: Output dimension
        """
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(i_dim, o_dim),
            nn.GELU(),
            nn.Linear(o_dim, o_dim),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through MLP.

        Args:
            x: Input features (B, T, i_dim)

        Returns:
            Projected features (B, T, o_dim)
        """
        return self.mlp(x)
