"""Composite loss functions for training denoising models, including Charbonnier, Wavelet, and SSIM terms."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Any, Dict, Tuple


class CharbonnierLoss(nn.Module):
    """Robust L1 alternative that prevents unstable gradients near zero."""

    def __init__(self, eps: float = 1e-3) -> None:
        """Initializes CharbonnierLoss.

        Args:
            eps: Epsilon value for stability. Defaults to 1e-3.
        """
        super().__init__()
        self.eps = eps * eps

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Computes the Charbonnier loss.

        Args:
            x: Predicted tensor.
            y: Target tensor.

        Returns:
            The computed loss.
        """
        diff = x - y
        loss = torch.sqrt(diff * diff + self.eps)
        return loss.mean()


class WaveletLoss(nn.Module):
    """Penalizes errors in frequency sub-bands using Haar wavelets."""

    def __init__(self) -> None:
        """Initializes WaveletLoss with Haar wavelet filters."""
        super().__init__()
        # Define Haar wavelet filters manually for fast, differentiable execution
        h = torch.tensor([[0.5, 0.5], [0.5, 0.5]]).view(1, 1, 2, 2)
        lh = torch.tensor([[-0.5, -0.5], [0.5, 0.5]]).view(1, 1, 2, 2)
        hl = torch.tensor([[-0.5, 0.5], [-0.5, 0.5]]).view(1, 1, 2, 2)
        hh = torch.tensor([[0.5, -0.5], [-0.5, 0.5]]).view(1, 1, 2, 2)

        self.register_buffer("filters", torch.cat([h, lh, hl, hh], dim=0))

    def _dwt(self, x: torch.Tensor) -> torch.Tensor:
        """Performs Discrete Wavelet Transform.

        Args:
            x: Input tensor of shape (B, C, H, W).

        Returns:
            DWT coefficients of shape (B, C, 4, H // 2, W // 2).
        """
        B, C, H, W = x.shape
        x_unfolded = F.conv2d(
            x.reshape(B * C, 1, H, W), self.filters, stride=2, padding=0
        )
        return x_unfolded.reshape(B, C, 4, H // 2, W // 2)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Computes the Wavelet loss.

        Args:
            x: Predicted tensor.
            y: Target tensor.

        Returns:
            The computed loss.
        """
        pad_w = x.shape[-1] % 2
        pad_h = x.shape[-2] % 2
        if pad_w != 0 or pad_h != 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
            y = F.pad(y, (0, pad_w, 0, pad_h), mode="reflect")
        return F.l1_loss(self._dwt(x), self._dwt(y))


class SSIMLoss(nn.Module):
    """Structural Similarity Index Measure (SSIM) Loss."""

    def __init__(
        self,
        window_size: int = 11,
        sigma: float = 1.5,
        channels: int = 3,
        size_average: bool = True,
    ) -> None:
        """Initializes SSIMLoss.

        Args:
            window_size: Size of the Gaussian window. Defaults to 11.
            sigma: Standard deviation of the Gaussian window. Defaults to 1.5.
            channels: Number of input channels. Defaults to 3.
            size_average: Whether to average the loss over the batch. Defaults to True.
        """
        super().__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channels = channels
        self.sigma = sigma
        window = self.create_window(window_size, sigma, channels)
        self.register_buffer("window", window)

    def gaussian(self, window_size: int, sigma: float) -> torch.Tensor:
        """Generates a 1D Gaussian kernel.

        Args:
            window_size: Size of the kernel.
            sigma: Standard deviation of the Gaussian.

        Returns:
            1D Gaussian kernel tensor.
        """
        gauss = torch.Tensor(
            [
                np.exp(-((x - window_size // 2) ** 2) / float(2 * sigma**2))
                for x in range(window_size)
            ]
        )
        return gauss / gauss.sum()

    def create_window(
        self, window_size: int, sigma: float, channel: int
    ) -> torch.Tensor:
        """Creates a 2D Gaussian window.

        Args:
            window_size: Size of the window.
            sigma: Standard deviation of the Gaussian.
            channel: Number of input channels.

        Returns:
            4D Gaussian window tensor (channel, 1, window_size, window_size).
        """
        _1D_window = self.gaussian(window_size, sigma).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window

    def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        """Computes the SSIM loss.

        Args:
            img1: First image tensor.
            img2: Second image tensor.

        Returns:
            The computed loss (1 - SSIM).
        """
        mu1 = F.conv2d(
            img1, self.window, padding=self.window_size // 2, groups=self.channels
        )
        mu2 = F.conv2d(
            img2, self.window, padding=self.window_size // 2, groups=self.channels
        )

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = (
            F.conv2d(
                img1 * img1,
                self.window,
                padding=self.window_size // 2,
                groups=self.channels,
            )
            - mu1_sq
        )
        sigma2_sq = (
            F.conv2d(
                img2 * img2,
                self.window,
                padding=self.window_size // 2,
                groups=self.channels,
            )
            - mu2_sq
        )
        sigma12 = (
            F.conv2d(
                img1 * img2,
                self.window,
                padding=self.window_size // 2,
                groups=self.channels,
            )
            - mu1_mu2
        )

        C1 = 0.01**2
        C2 = 0.03**2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
            (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
        )

        if self.size_average:
            return 1 - ssim_map.mean()
        else:
            return 1 - ssim_map.mean(dim=(1, 2, 3))


class CompositeLoss(nn.Module):
    """Combines Charbonnier, Wavelet and SSIM domain constraints."""

    def __init__(self, config: Any) -> None:
        """Initializes CompositeLoss.

        Args:
            config: Configuration object containing loss weights and other settings.
        """
        super().__init__()
        self.config = config
        self.charbonnier = CharbonnierLoss()
        self.wavelet = WaveletLoss()
        self.ssim = (
            SSIMLoss(channels=config.out_channels)
            if config.ssim_weight != 0.0
            else None
        )

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Computes the combined composite loss.

        Args:
            pred: Predicted tensor.
            target: Target tensor.

        Returns:
            A tuple containing:
                - The total balanced loss.
                - A dictionary containing individual loss components.
        """
        l_char = self.charbonnier(pred, target)
        l_wave = self.wavelet(pred, target)
        if self.ssim is not None:
            l_ssim = self.ssim(pred, target)
        else:
            l_ssim = pred.new_tensor(0.0)

        total_loss = (
            self.config.charbonnier_weight * l_char
            + self.config.wavelet_weight * l_wave
            + self.config.ssim_weight * l_ssim
        )
        return total_loss, {
            "loss_char": l_char,
            "loss_wave": l_wave,
            "loss_ssim": l_ssim,
        }
