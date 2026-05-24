import torch
import torch.nn as nn

from models.lonpe import LoNPE
from models.mamba_ir import AttentiveStateSpaceBlock
from models.nafnet import NAFBlock


class HASST(nn.Module):
    """
    Hybrid Attentive State-Space Transformer (HASST)
    Combines Local CNNs, Global State Space Models, and Sensor Conditioners.
    """

    def __init__(self, in_channels=3, out_channels=3, embed_dim=64, num_blocks=4):
        super().__init__()

        # 1. Noise Conditioning Prior Module (Outputs 2 channels: shot, read)
        self.lonpe = LoNPE(in_channels=in_channels, out_channels=2)

        # 2. Shallow Feature Extraction
        self.intro = nn.Conv2d(in_channels, embed_dim, kernel_size=3, padding=1)

        # 3. Deep Hybrid Restoration Feature Pipeline
        # Interleaved blocks: Local (NAF) then Global (Mamba ASSB)
        self.local_blocks = nn.ModuleList(
            [NAFBlock(embed_dim) for _ in range(num_blocks)]
        )
        self.global_blocks = nn.ModuleList(
            [AttentiveStateSpaceBlock(embed_dim) for _ in range(num_blocks)]
        )

        # 4. Reconstruction Output Block
        self.ending = nn.Conv2d(embed_dim, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        # Global residual hook
        residual_identity = x

        # Extract sensor-level local noise prior maps (2 channels)
        noise_prior = self.lonpe(x)

        # Map input image to latent embedding space
        feat = self.intro(x)

        # Interleaved Sequential Processing (Local -> Global)
        # Information flows through local refinement then global context extraction
        for i in range(len(self.local_blocks)):
            feat = self.local_blocks[i](feat, noise_prior)
            feat = self.global_blocks[i](feat, noise_prior)

        # Map back to standard sRGB image dimensions
        out = self.ending(feat) + residual_identity

        if self.training:
            return out
        else:
            return torch.clamp(out, 0.0, 1.0)
