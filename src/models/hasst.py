"""Hybrid Attentive State-Space Transformer (HASST) model architecture for image denoising."""

import torch
import torch.nn as nn
from typing import Optional, Tuple

from models.lonpe import LoNPE
from models.mamba_ir import AttentiveStateSpaceBlock
from models.nafnet import NAFBlock


class BCU(nn.Module):
    """Bidirectional Connection Unit for sensor-aware local/global fusion.

    This is intentionally lightweight for fast training:
    - only 1x1 convolutions are used inside the fusion path
    - noise prior is embedded once and reused for both branches
    - refinement is explicitly bidirectional: local refines global and global refines local
    """

    def __init__(
        self, embed_dim: int, prior_channels: int = 2, reduction: int = 4
    ) -> None:
        """Initializes BCU.

        Args:
            embed_dim: Dimension of the input features.
            prior_channels: Number of channels in the noise prior.
            reduction: Reduction factor for the hidden dimension.
        """
        super().__init__()
        hidden_dim = max(embed_dim // reduction, 8)

        self.noise_embed = nn.Sequential(
            nn.Conv2d(prior_channels, hidden_dim, kernel_size=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_dim, embed_dim, kernel_size=1),
        )

        self.local_gate = nn.Sequential(
            nn.Conv2d(embed_dim * 2, hidden_dim, kernel_size=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_dim, embed_dim, kernel_size=1),
            nn.Sigmoid(),
        )
        self.global_gate = nn.Sequential(
            nn.Conv2d(embed_dim * 2, hidden_dim, kernel_size=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_dim, embed_dim, kernel_size=1),
            nn.Sigmoid(),
        )

        self.mix_logits = nn.Conv2d(embed_dim * 3, 2, kernel_size=1)
        self.out_proj = nn.Conv2d(embed_dim, embed_dim, kernel_size=1)

        self.res_scale = nn.Parameter(torch.ones(1, embed_dim, 1, 1) * 1e-2)

    def forward(
        self,
        f_local: torch.Tensor,
        f_global: torch.Tensor,
        noise_prior: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            f_local: Local features.
            f_global: Global features.
            noise_prior: Optional noise prior map.

        Returns:
            Fused features.
        """
        if noise_prior is None:
            noise_feat = torch.zeros_like(f_local)
        else:
            noise_feat = self.noise_embed(noise_prior)

        local_to_global = self.local_gate(torch.cat([f_local, noise_feat], dim=1))
        global_to_local = self.global_gate(torch.cat([f_global, noise_feat], dim=1))

        local_refined = f_local + global_to_local * f_global
        global_refined = f_global + local_to_global * f_local

        mix = torch.softmax(
            self.mix_logits(
                torch.cat([local_refined, global_refined, noise_feat], dim=1)
            ),
            dim=1,
        )
        fused = mix[:, 0:1] * local_refined + mix[:, 1:2] * global_refined
        return f_local + self.out_proj(fused) * self.res_scale


class HASSTBlock(nn.Module):
    """Hybrid Attentive State-Space Block (HASSTBlock).

    Combines a local NAFBlock and a global AttentiveStateSpaceBlock with a BCU.
    """

    def __init__(self, c: int) -> None:
        """Initializes HASSTBlock.

        Args:
            c: Number of input/output channels.
        """
        super().__init__()
        self.l = NAFBlock(c)
        self.g = AttentiveStateSpaceBlock(c)
        self.bcu = BCU(c)

    def forward(
        self, feat: torch.Tensor, noise_prior: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            feat: Input features.
            noise_prior: Optional noise prior map.

        Returns:
            Output features.
        """
        if noise_prior is not None and noise_prior.shape[2:] != feat.shape[2:]:
            noise_prior_down = nn.functional.interpolate(
                noise_prior, size=feat.shape[2:], mode="bilinear", align_corners=False
            )
        else:
            noise_prior_down = noise_prior

        local_feat = self.l(feat, noise_prior_down)
        global_feat = self.g(feat, noise_prior_down)
        return self.bcu(local_feat, global_feat, noise_prior_down)


class HASST(nn.Module):
    """Hybrid Attentive State-Space Transformer (HASST).

    Multi-scale U-Net architecture combining CNN and Mamba-based blocks.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        embed_dim: int = 64,
        num_blocks: int = 4,
        lonpe_scale_physical: bool = True,
        lonpe_shot_range: Tuple[float, float] = (1.0e-5, 5.0e-1),
        lonpe_read_range: Tuple[float, float] = (1.0e-6, 1.0e-2),
    ) -> None:
        """Initializes HASST.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            embed_dim: Number of base embedding channels.
            num_blocks: Total number of HASST blocks to distribute across scales.
            lonpe_scale_physical: Whether LoNPE should scale to physical ranges.
            lonpe_shot_range: Range for shot noise estimation.
            lonpe_read_range: Range for read noise estimation.
        """
        super().__init__()

        # 1. Noise Conditioning Prior Module (Outputs 2 channels: shot, read)
        self.lonpe = LoNPE(
            in_channels=in_channels,
            out_channels=2,
            scale_physical=lonpe_scale_physical,
            shot_range=lonpe_shot_range,
            read_range=lonpe_read_range,
        )

        # 2. Shallow Feature Extraction
        self.intro = nn.Conv2d(in_channels, embed_dim, kernel_size=3, padding=1)

        # 3. U-Net structure
        num_scales = 4
        mid_blocks = 2
        remaining = max(0, num_blocks - mid_blocks)
        if remaining % 2 == 1:
            # Keep encoder/decoder symmetric: their total must be even.
            mid_blocks += 1
            remaining -= 1

        enc_total = remaining // 2
        base_per_stage = enc_total // num_scales
        stage_remainder = enc_total % num_scales
        blk_nums = [
            base_per_stage + (1 if i < stage_remainder else 0)
            for i in range(num_scales)
        ]

        self.enc = nn.ModuleList()
        self.down = nn.ModuleList()
        c = embed_dim
        for num in blk_nums:
            self.enc.append(nn.ModuleList([HASSTBlock(c) for _ in range(num)]))
            self.down.append(nn.Conv2d(c, c * 2, kernel_size=2, stride=2))
            c *= 2

        self.mid = nn.ModuleList([HASSTBlock(c) for _ in range(mid_blocks)])

        self.up = nn.ModuleList()
        self.dec = nn.ModuleList()
        for num in reversed(blk_nums):
            self.up.append(nn.ConvTranspose2d(c, c // 2, kernel_size=2, stride=2))
            c //= 2
            self.dec.append(
                nn.ModuleList(
                    [
                        nn.Conv2d(
                            c * 2, c, kernel_size=1
                        ),  # feature reduction for concats
                        *[HASSTBlock(c) for _ in range(num)],
                    ]
                )
            )

        # 4. Reconstruction Output Block
        self.ending = nn.Conv2d(embed_dim, out_channels, kernel_size=3, padding=1)

        # Zero-initialize the global residual so the network starts as an identity function
        nn.init.zeros_(self.ending.weight)
        nn.init.zeros_(self.ending.bias)

    def estimate_noise_prior(self, x: torch.Tensor) -> torch.Tensor:
        """Estimates the noise prior from the input image.

        Args:
            x: Input image tensor.

        Returns:
            Noise prior map.
        """
        return self.lonpe(x)

    def forward(
        self, x: torch.Tensor, noise_prior: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input image tensor of shape (B, C, H, W).
            noise_prior: Optional noise prior map. If None, it is estimated using LoNPE.

        Returns:
            Denoised image tensor.
        """
        # Global residual hook
        residual_identity = x

        # Extract sensor-level local noise prior maps (2 channels)
        if noise_prior is None:
            noise_prior = self.estimate_noise_prior(x)

        # Map input image to latent embedding space
        feat = self.intro(x)

        skips = []
        for enc_blocks, down in zip(self.enc, self.down):
            for blk in enc_blocks:
                feat = blk(feat, noise_prior)
            skips.append(feat)
            feat = down(feat)

        for blk in self.mid:
            feat = blk(feat, noise_prior)

        for up, dec_blocks, skip in zip(self.up, self.dec, reversed(skips)):
            feat = up(feat)
            feat = torch.cat([feat, skip], dim=1)
            feat = dec_blocks[0](feat)  # reduce channels
            for blk in dec_blocks[1:]:
                feat = blk(feat, noise_prior)

        # Map back to standard sRGB image dimensions
        out = self.ending(feat) + residual_identity

        if self.training:
            return out
        else:
            return torch.clamp(out, 0.0, 1.0)
