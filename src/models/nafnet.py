"""Nonlinear Activation Free Network (NAFNet) blocks with noise conditioning support."""

import torch
import torch.nn as nn
from typing import Optional


class LayerNorm2d(nn.Module):
    """Channel-wise Layer Normalization for 4D (BCHW) tensors.

    Mathematically identical to nn.LayerNorm but avoids expensive permutations.
    """

    def __init__(self, channels: int, eps: float = 1e-6) -> None:
        """Initializes LayerNorm2d.

        Args:
            channels: Number of channels in the input tensor.
            eps: A value added to the denominator for numerical stability.
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (B, C, H, W).

        Returns:
            Normalized tensor.
        """
        mean = x.mean(1, keepdim=True)
        var = x.var(1, keepdim=True, unbiased=False)
        x = (x - mean) / torch.sqrt(var + self.eps)
        return x * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)


class SimpleGate(nn.Module):
    """Core NAFNet innovation: nonlinear multiplication of feature chunks."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (B, C, H, W).

        Returns:
            Gated tensor of shape (B, C//2, H, W).
        """
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class NAFBlock(nn.Module):
    """Nonlinear Activation Free Block (NAFBlock).

    Official Megvii-Research implementation structure.
    Extended with LoNPE-based conditional modulation.
    """

    def __init__(self, c: int, dw_expand: int = 2, ffn_expand: int = 2) -> None:
        """Initializes NAFBlock.

        Args:
            c: Number of input/output channels.
            dw_expand: Expansion factor for depthwise convolution.
            ffn_expand: Expansion factor for feed-forward network.
        """
        super().__init__()
        dw_channel = c * dw_expand
        self.conv1 = nn.Conv2d(c, dw_channel, kernel_size=1, bias=True)
        self.conv2 = nn.Conv2d(
            dw_channel,
            dw_channel,
            kernel_size=3,
            padding=1,
            groups=dw_channel,
            bias=True,
        )
        self.conv3 = nn.Conv2d(dw_channel // 2, c, kernel_size=1, bias=True)

        # Simplified Channel Attention (SCA)
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dw_channel // 2, dw_channel // 2, kernel_size=1, bias=True),
        )

        # Initialize SCA bias to 1.0 to allow signal pass-through at training start
        nn.init.constant_(self.sca[1].bias, 1.0)

        self.sg = SimpleGate()

        ffn_channel = c * ffn_expand
        self.conv4 = nn.Conv2d(c, ffn_channel, kernel_size=1, bias=True)
        self.conv5 = nn.Conv2d(ffn_channel // 2, c, kernel_size=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.beta = nn.Parameter(torch.ones(c) * 1e-2, requires_grad=True)
        self.gamma = nn.Parameter(torch.ones(c) * 1e-2, requires_grad=True)

        # Conditional Modulation from LoNPE (2 channels: shot, read)
        self.cond_proj = nn.Sequential(nn.Conv2d(2, c, kernel_size=1), nn.Sigmoid())

    def forward(
        self, inp: torch.Tensor, noise_prior: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            inp: Input tensor of shape (B, C, H, W).
            noise_prior: Optional noise prior tensor of shape (B, 2, H, W).

        Returns:
            Output tensor of shape (B, C, H, W).
        """
        # Apply conditional modulation if prior is provided (Centered around 1.0)
        x_in = inp
        if noise_prior is not None:
            cond_scale = self.cond_proj(noise_prior)
            x_in = x_in * (1 + cond_scale)

        # 1. Spatial / Attention Branch
        x = self.norm1(x_in)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)
        y = inp + x * self.beta.view(1, -1, 1, 1)

        # 2. Feed-forward / Channel Branch
        x = self.norm2(y)
        x = self.conv4(x)
        x = self.sg(x)
        x = self.conv5(x)
        return y + x * self.gamma.view(1, -1, 1, 1)
