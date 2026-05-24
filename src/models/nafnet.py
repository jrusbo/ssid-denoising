import torch
import torch.nn as nn


class SimpleGate(nn.Module):
    """Core NAFNet innovation: nonlinear multiplication of feature chunks."""

    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class NAFBlock(nn.Module):
    """
    Nonlinear Activation Free Block (NAFBlock).
    Official Megvii-Research implementation structure.
    Extended with LoNPE-based conditional modulation.
    """

    def __init__(self, c, dw_expand=2, ffn_expand=2):
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

        self.sg = SimpleGate()

        ffn_channel = c * ffn_expand
        self.conv4 = nn.Conv2d(c, ffn_channel, kernel_size=1, bias=True)
        self.conv5 = nn.Conv2d(ffn_channel // 2, c, kernel_size=1, bias=True)

        self.norm1 = nn.LayerNorm(c)
        self.norm2 = nn.LayerNorm(c)

        self.beta = nn.Parameter(torch.ones((1, c, 1, 1)) * 1e-2, requires_grad=True)
        self.gamma = nn.Parameter(torch.ones((1, c, 1, 1)) * 1e-2, requires_grad=True)

        # Conditional Modulation from LoNPE (2 channels: shot, read)
        self.cond_proj = nn.Sequential(
            nn.Conv2d(2, c, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, inp, noise_prior=None):
        # Apply conditional modulation if prior is provided (Centered around 1.0)
        x_in = inp
        if noise_prior is not None:
            cond_scale = self.cond_proj(noise_prior)
            x_in = x_in * (1 + cond_scale)

        # 1. Spatial / Attention Branch
        x = self.norm1(x_in.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)
        y = inp + x * self.beta

        # 2. Feed-forward / Channel Branch
        x = self.norm2(y.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x = self.conv4(x)
        x = self.sg(x)
        x = self.conv5(x)
        return y + x * self.gamma
