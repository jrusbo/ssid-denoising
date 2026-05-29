import torch
import torch.nn as nn
from einops import rearrange

# Using the officially maintained Mamba implementation
try:
    from mamba_ssm.modules.mamba_simple import Mamba
except ImportError:
    import warnings
    warnings.warn("mamba_ssm not found. AttentiveStateSpaceBlock will fallback to Zero for the global branch. "
                  "Please install mamba-ssm and causal-conv1d.")
    Mamba = None


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
    Upgraded to strict MambaIRv2 standards with ASE and SGN.
    Extended with LoNPE-based conditional modulation.
    """

    def __init__(self, c, d_state=16, d_conv=4, expand=2, num_prompts=64):
        super().__init__()
        self.norm1 = nn.LayerNorm(c)
        self.norm2 = nn.LayerNorm(c)
        self.num_prompts = num_prompts

        # ASE: Learnable prompt parameter
        self.prompts = nn.Parameter(torch.randn(1, num_prompts, c) * 0.02)

        # SGN: Semantic guidance projection
        self.sgn_proj = nn.Linear(c, 1)

        # Mamba acts as the non-causal token mixer
        if Mamba is not None:
            self.mamba = Mamba(d_model=c, d_state=d_state, d_conv=d_conv, expand=expand)
        else:
            self.mamba = None

        self.ffn = MambaFFN(c)

        # Stabilization parameters
        self.mamba_beta = nn.Parameter(torch.ones((1, c, 1, 1)) * 1e-2, requires_grad=True)
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

        # Global Branch: Mamba Token Mixer with SGN and ASE
        if self.mamba is not None:
            # SGN: Semantic Guided Neighboring
            sem_labels = self.sgn_proj(x_norm_seq).squeeze(-1) # (B, L)
            sort_idx = torch.argsort(sem_labels, dim=-1) # (B, L)
            restore_idx = torch.argsort(sort_idx, dim=-1) # (B, L)

            # Gather based on semantic similarity
            idx_expanded = sort_idx.unsqueeze(-1).expand(-1, -1, C)
            x_sgn = torch.gather(x_norm_seq, 1, idx_expanded)

            # ASE: Attentive State-space Equation (Prompting)
            prompts_repeated = self.prompts.expand(B, -1, -1)
            x_prompted = torch.cat([prompts_repeated, x_sgn], dim=1) # (B, num_prompts + L, C)

            # Mamba Scan
            x_m_prompted = self.mamba(x_prompted)

            # Remove Prompts
            x_m_sgn = x_m_prompted[:, self.num_prompts:, :]

            # Restore Original Sequence Order
            restore_idx_expanded = restore_idx.unsqueeze(-1).expand(-1, -1, C)
            x_m_seq = torch.gather(x_m_sgn, 1, restore_idx_expanded)

            x_m = rearrange(x_m_seq, "b (h w) c -> b c h w", h=H, w=W)
        else:
            x_m = torch.zeros_like(x_in)

        # Combine with residual and stabilization scales
        x = x + x_m * self.mamba_beta

        # Feed-forward Branch
        res = x
        x_f_seq = rearrange(x, "b c h w -> b (h w) c")
        x_f = self.ffn(self.norm2(x_f_seq))
        x_f = rearrange(x_f, "b (h w) c -> b c h w", h=H, w=W)
        return res + x_f * self.ffn_gamma
