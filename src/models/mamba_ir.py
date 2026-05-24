import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

# Using the officially maintained Mamba implementation
try:
    from mamba_ssm.modules.mamba_simple import Mamba
except ImportError:
    import warnings
    warnings.warn("mamba_ssm not found. AttentiveStateSpaceBlock will fallback to Zero for the global branch. "
                  "Please install mamba-ssm and causal-conv1d.")
    Mamba = None


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.reshape(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = (
        x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    )
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.reshape(
        B, H // window_size, W // window_size, window_size, window_size, -1
    )
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    """Window based multi-head self attention (W-MSA) module."""

    def __init__(self, dim, window_size, num_heads, qkv_bias=True):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
        """
        B_, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B_, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        return x


class MambaFFN(nn.Module):
    """Standard FFN for State-Space Transformers."""

    def __init__(self, c, expand=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(c, c * expand),
            nn.GELU(),
            nn.Linear(c * expand, c),
        )

    def forward(self, x):
        return self.net(x)


class AttentiveStateSpaceBlock(nn.Module):
    """
    Attentive State Space Block (ASSB) for MambaIRv2.
    Official Transformer-like structure: Norm -> Parallel(Mamba, WindowAttention) -> Norm -> FFN.
    Extended with LoNPE-based conditional modulation.
    """

    def __init__(self, c, d_state=16, d_conv=4, expand=2, window_size=8, num_heads=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(c)
        self.norm2 = nn.LayerNorm(c)
        self.window_size = window_size

        # Mamba acts as the non-causal token mixer
        if Mamba is not None:
            self.mamba = Mamba(d_model=c, d_state=d_state, d_conv=d_conv, expand=expand)
        else:
            self.mamba = None

        # Window Attention acts as the local feature mixer
        self.window_attn = WindowAttention(
            dim=c, window_size=window_size, num_heads=num_heads
        )

        self.ffn = MambaFFN(c)

        # Stabilization parameters
        self.mamba_beta = nn.Parameter(torch.ones((1, c, 1, 1)) * 1e-2, requires_grad=True)
        self.attn_beta = nn.Parameter(torch.ones((1, c, 1, 1)) * 1e-2, requires_grad=True)
        self.ffn_gamma = nn.Parameter(torch.ones((1, c, 1, 1)) * 1e-2, requires_grad=True)

        # Conditional Modulation from LoNPE (2 channels: shot, read)
        self.cond_proj = nn.Sequential(
            nn.Conv2d(2, c, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x, noise_prior=None):
        B, C, H, W = x.shape

        # Apply conditional modulation if prior is provided (Centered around 1.0)
        x_in = x
        if noise_prior is not None:
            cond_scale = self.cond_proj(noise_prior)
            x_in = x_in * (1 + cond_scale)

        # Normalization for mixers
        x_seq = rearrange(x_in, "b c h w -> b (h w) c")
        x_norm_seq = self.norm1(x_seq)
        x_norm = rearrange(x_norm_seq, "b (h w) c -> b h w c", h=H, w=W)

        # 1. Global Branch: Mamba Token Mixer
        if self.mamba is not None:
            x_m = self.mamba(x_norm_seq)
            x_m = rearrange(x_m, "b (h w) c -> b c h w", h=H, w=W)
        else:
            x_m = torch.zeros_like(x_in)

        # 2. Local Branch: Window Attention
        # Partition windows
        pad_h = (self.window_size - H % self.window_size) % self.window_size
        pad_w = (self.window_size - W % self.window_size) % self.window_size
        x_norm_pad = F.pad(x_norm, (0, 0, 0, pad_w, 0, pad_h))
        Hp, Wp = x_norm_pad.shape[1], x_norm_pad.shape[2]

        x_windows = window_partition(
            x_norm_pad, self.window_size
        )  # (nW*B, window_size, window_size, C)
        x_windows = x_windows.reshape(-1, self.window_size * self.window_size, C)

        # W-MSA
        attn_windows = self.window_attn(x_windows)

        # Merge windows
        attn_windows = attn_windows.reshape(-1, self.window_size, self.window_size, C)
        x_attn = window_reverse(attn_windows, self.window_size, Hp, Wp)

        # Unpad
        if pad_h > 0 or pad_w > 0:
            x_attn = x_attn[:, :H, :W, :].contiguous()

        x_attn = rearrange(x_attn, "b h w c -> b c h w")

        # Combine parallel mixers with residual and stabilization scales
        x = x + x_m * self.mamba_beta + x_attn * self.attn_beta

        # 3. Feed-forward Branch
        res = x
        x_f_seq = rearrange(x, "b c h w -> b (h w) c")
        x_f = self.ffn(self.norm2(x_f_seq))
        x_f = rearrange(x_f, "b (h w) c -> b c h w", h=H, w=W)
        return res + x_f * self.ffn_gamma
