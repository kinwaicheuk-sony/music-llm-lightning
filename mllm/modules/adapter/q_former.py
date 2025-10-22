"""
Q-Former adapter with attention-based learnable queries.
"""
import torch
from torch import nn
from einops import repeat, rearrange
from dataclasses import dataclass
from typing import Optional

def exists(val) -> bool:
    """Check if value is not None."""
    return val is not None

def default(val, d):
    """Return val if exists, else return default d."""
    return val if exists(val) else d

class PreNorm(nn.Module):
    """Layer normalization wrapper for attention/feedforward blocks."""

    def __init__(self, dim: int, fn: nn.Module, context_dim: int = None):
        """
        Initialize prenorm wrapper.

        Args:
            dim: Input dimension
            fn: Function/module to wrap
            context_dim: Context dimension for cross-attention
        """
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)
        self.norm_context = nn.LayerNorm(context_dim) if exists(context_dim) else None

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Apply normalization and forward through wrapped function."""
        x = self.norm(x)
        if exists(self.norm_context):
            context = kwargs['context']
            normed_context = self.norm_context(context)
            kwargs.update(context = normed_context)
        return self.fn(x, **kwargs)

class GEGLU(nn.Module):
    """Gated GELU activation function."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply gated GELU activation."""
        x, gates = x.chunk(2, dim = -1)
        return x * nn.functional.gelu(gates)

class FeedForward(nn.Module):
    """Feedforward network with GEGLU activation."""

    def __init__(self, dim: int, mult: int = 4):
        """
        Initialize feedforward network.

        Args:
            dim: Input/output dimension
            mult: Hidden dimension multiplier
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Linear(dim * mult, dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward through feedforward network."""
        return self.net(x)

class Attention(nn.Module):
    """Multi-head cross-attention mechanism."""

    def __init__(self, query_dim: int, context_dim: int = None, heads: int = 8, dim_head: int = 64):
        """
        Initialize attention module.

        Args:
            query_dim: Query dimension
            context_dim: Context dimension (defaults to query_dim)
            heads: Number of attention heads
            dim_head: Dimension per head
        """
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)
        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias = False)
        self.to_out = nn.Linear(inner_dim, query_dim, bias = False)

    def forward(self, x: torch.Tensor, context: torch.Tensor = None, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Apply cross-attention.

        Args:
            x: Query tensor (B, N, D)
            context: Context tensor for cross-attention
            mask: Attention mask

        Returns:
            Attention output (B, N, D)
        """
        h = self.heads
        q = self.to_q(x)
        context = default(context, x)
        k, v = self.to_kv(context).chunk(2, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h = h), (q, k, v))

        with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=True, enable_mem_efficient=True):
            out = nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask)

        out = rearrange(out, '(b h) n d -> b n (h d)', h = h)
        return self.to_out(out)

class AL_Adapter(nn.Module):
    """Attention-based Learnable adapter with cross-attention layers."""

    def __init__(self, num_learnable_latents: int, num_layers: int, i_dim: int, o_dim: int, num_heads: int):
        """
        Initialize AL_Adapter.

        Args:
            num_learnable_latents: Number of learnable query tokens
            num_layers: Number of cross-attention layers
            i_dim: Input dimension
            o_dim: Output dimension
            num_heads: Number of attention heads
        """
        super().__init__()

        self.num_layers = num_layers
        self.num_learnable_latents = num_learnable_latents
        self.latent_dim = o_dim
        self.i_dim = i_dim
        self.o_dim = o_dim
        self.cross_heads = num_heads
        self.cross_attend_blocks = nn.ModuleList([
            PreNorm(self.latent_dim,
                   Attention(self.latent_dim, self.o_dim, heads=self.cross_heads),
                   context_dim=self.o_dim),
            PreNorm(self.latent_dim, FeedForward(self.latent_dim))
        ])
        self.layers = nn.ModuleList([self.cross_attend_blocks] * self.num_layers)
        self.latents = nn.Parameter(torch.randn(self.num_learnable_latents, self.latent_dim))
        if self.i_dim != self.o_dim:
            self.projection = nn.Linear(self.i_dim, self.o_dim, bias=False)
        else:
            self.projection = nn.Identity()

    def forward(self, audio_embedding: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through adapter.

        Args:
            audio_embedding: Input audio features (B, T, i_dim)
            attention_mask: Optional attention mask

        Returns:
            Adapted features (B, num_learnable_latents, o_dim)
        """
        batch_size = audio_embedding.shape[0]
        x = repeat(self.latents, 'n d -> b n d', b=batch_size)
        audio_embedding = self.projection(audio_embedding)
        for cross_attn, cross_ff in self.layers:
            x = cross_attn(x, context=audio_embedding, mask=attention_mask) + x
            x = cross_ff(x) + x
        return x
