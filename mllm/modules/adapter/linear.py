"""
Linear adapter for projecting audio features to LLM space.
"""
import torch
import torch.nn as nn

class LinearAdapter(nn.Module):
    """Simple linear projection adapter."""

    def __init__(self, i_dim: int, o_dim: int):
        """
        Initialize linear adapter.

        Args:
            i_dim: Input dimension
            o_dim: Output dimension
        """
        super().__init__()
        self.linear = nn.Linear(i_dim, o_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through linear projection.

        Args:
            x: Input features (B, T, i_dim)

        Returns:
            Projected features (B, T, o_dim)
        """
        return self.linear(x)
