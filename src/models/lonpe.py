import torch
import torch.nn as nn
from typing import Tuple


class LoNPE(nn.Module):
    """Locally Noise Prior Estimation (LoNPE) Module.

    Official implementation as per Condformer (IJCV 2025).
    Estimates a 2-channel dense spatial noise prior map (shot noise, read noise) from the noisy input.

    Behavior changes implemented here:
    - The network head retains a Sigmoid output in [0, 1] but we optionally map
      those normalized estimates to physically meaningful ranges for shot and read
      noise. This keeps backward compatibility (returns a 2-channel tensor) while
      providing a scaled interpretation useful for conditioning and diagnostics.
    """

    def __init__(
        self,
        in_channels: int = 3,
        mid_channels: int = 32,
        out_channels: int = 2,
        shot_range: Tuple[float, float] = (1e-5, 5e-1),
        read_range: Tuple[float, float] = (1e-6, 1e-2),
        scale_physical: bool = True,
    ) -> None:
        """Initializes the LoNPE module.

        Args:
            in_channels: Number of input channels.
            mid_channels: Number of intermediate channels.
            out_channels: Number of output channels (typically 2 for shot and read noise).
            shot_range: Minimum and maximum for the shot-noise scale.
            read_range: Minimum and maximum for the read-noise variance.
            scale_physical: If True, maps sigmoid outputs to physical ranges.
        """
        super(LoNPE, self).__init__()
        # Official architecture: 3x3 Conv -> ReLU -> 3x3 Conv -> ReLU -> 3x3 Conv -> Sigmoid
        self.estimation = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

        # Physical ranges and behaviour flag
        self.shot_min, self.shot_max = float(shot_range[0]), float(shot_range[1])
        self.read_min, self.read_max = float(read_range[0]), float(read_range[1])
        self.scale_physical = bool(scale_physical)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (B, C, H, W).

        Returns:
            Noise prior map of shape (B, 2, H, W).
        """
        s = self.estimation(x)  # (B,2,H,W) in [0,1]

        if not self.scale_physical:
            return s

        # Map sigmoid outputs to physically plausible ranges per-channel:
        # channel 0 -> shot noise scale (e.g., proportional to signal),
        # channel 1 -> read noise variance
        shot = s[:, 0:1, ...] * (self.shot_max - self.shot_min) + self.shot_min
        read = s[:, 1:2, ...] * (self.read_max - self.read_min) + self.read_min

        return torch.cat([shot, read], dim=1)
