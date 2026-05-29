import torch
import torch.nn as nn

from models.lonpe import LoNPE
from models.mamba_ir import AttentiveStateSpaceBlock
from models.nafnet import NAFBlock


class BCU(nn.Module):
    """Bidirectional Connection Unit for feature fusion"""
    def __init__(self, embed_dim):
        super().__init__()
        self.cond_proj = nn.Sequential(
            nn.Conv2d(2, embed_dim, kernel_size=1),
            nn.Sigmoid()
        )
        self.fusion = nn.Conv2d(embed_dim * 2, embed_dim, kernel_size=1)

    def forward(self, f_local, f_global, noise_prior):
        weight = self.cond_proj(noise_prior)
        f_local = f_local * weight
        f_global = f_global * (1 - weight)
        f_fused = torch.cat([f_local, f_global], dim=1)
        return self.fusion(f_fused)


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

        # Fusion Modules
        self.bcus = nn.ModuleList([BCU(embed_dim) for _ in range(num_blocks)])

        # 4. Reconstruction Output Block
        self.ending = nn.Conv2d(embed_dim, out_channels, kernel_size=3, padding=1)

        # Zero-initialize the global residual so the network starts as an identity function
        nn.init.zeros_(self.ending.weight)
        nn.init.zeros_(self.ending.bias)

    def forward(self, x):
        # Global residual hook
        residual_identity = x

        # Extract sensor-level local noise prior maps (2 channels)
        noise_prior = self.lonpe(x)

        # Map input image to latent embedding space
        feat = self.intro(x)

        # Parallel Branch Processing (Local || Global) with BCU Fusion
        for i in range(len(self.local_blocks)):
            local_feat = self.local_blocks[i](feat, noise_prior)
            global_feat = self.global_blocks[i](feat, noise_prior)

            feat = self.bcus[i](local_feat, global_feat, noise_prior)

        # Map back to standard sRGB image dimensions
        out = self.ending(feat) + residual_identity

        if self.training:
            return out
        else:
            return torch.clamp(out, 0.0, 1.0)
