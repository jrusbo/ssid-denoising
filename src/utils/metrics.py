"""Evaluation metrics for image denoising, providing GPU-accelerated PSNR and SSIM computations."""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Union


_ssim_window_cache: Dict[Tuple[int, int, float, str], torch.Tensor] = {}


@torch.no_grad()
def compute_psnr(
    pred: Union[torch.Tensor, np.ndarray], gt: Union[torch.Tensor, np.ndarray]
) -> torch.Tensor:
    """Computes Peak Signal-to-Noise Ratio (PSNR).

    Computes PSNR on GPU for speed. Assumes tensors are in the [0, 1] range.
    Can handle batched or single image tensors.

    Args:
        pred: Predicted image(s).
        gt: Ground truth image(s).

    Returns:
        The PSNR value as a torch Tensor. Returns the sum of PSNRs if batched.
    """
    if not isinstance(pred, torch.Tensor):
        pred = torch.tensor(pred)
    if not isinstance(gt, torch.Tensor):
        gt = torch.tensor(gt)

    mse = F.mse_loss(pred, gt, reduction="none")

    if mse.dim() == 4:
        # Batched input (B, C, H, W)
        mse = mse.reshape(mse.size(0), -1).mean(dim=1)
        psnr = 10 * torch.log10(
            torch.tensor(1.0, device=mse.device, dtype=mse.dtype) / (mse + 1e-10)
        )
        return psnr.sum()
    else:
        # Single image (C, H, W) or (H, W)
        mse = mse.mean()
        psnr = 10 * torch.log10(
            torch.tensor(1.0, device=mse.device, dtype=mse.dtype) / (mse + 1e-10)
        )
        return psnr


@torch.no_grad()
def compute_ssim(
    pred: torch.Tensor,
    gt: torch.Tensor,
    window_size: int = 11,
    sigma: float = 1.5,
    size_average: bool = True,
) -> torch.Tensor:
    """Computes Structural Similarity Index Measure (SSIM).

    Computes SSIM on GPU. Reference: https://github.com/Po-Hsun-Su/pytorch-ssim

    Args:
        pred: Predicted image(s).
        gt: Ground truth image(s).
        window_size: The size of the Gaussian window. Defaults to 11.
        sigma: The standard deviation of the Gaussian window. Defaults to 1.5.
        size_average: If True, returns the average SSIM over the batch.
            If False, returns the sum. Defaults to True.

    Returns:
        The SSIM value as a torch Tensor.
    """
    if pred.dim() == 3:
        pred = pred.unsqueeze(0)
        gt = gt.unsqueeze(0)

    device = pred.device
    channel = pred.size(1)

    cache_key = (window_size, channel, float(sigma), str(device))
    global _ssim_window_cache
    if cache_key not in _ssim_window_cache:
        _ssim_window_cache[cache_key] = _create_window(window_size, sigma, channel).to(
            device
        )

    window = _ssim_window_cache[cache_key]

    mu1 = F.conv2d(pred, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(gt, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = (
        F.conv2d(pred * pred, window, padding=window_size // 2, groups=channel) - mu1_sq
    )
    sigma2_sq = (
        F.conv2d(gt * gt, window, padding=window_size // 2, groups=channel) - mu2_sq
    )
    sigma12 = (
        F.conv2d(pred * gt, window, padding=window_size // 2, groups=channel) - mu1_mu2
    )

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(dim=(1, 2, 3)).sum()


def _gaussian(window_size: int, sigma: float) -> torch.Tensor:
    """Generates a 1D Gaussian kernel.

    Args:
        window_size: The size of the kernel.
        sigma: The standard deviation of the Gaussian.

    Returns:
        A 1D Gaussian kernel tensor.
    """
    gauss = torch.Tensor(
        [
            np.exp(-((x - window_size // 2) ** 2) / float(2 * sigma**2))
            for x in range(window_size)
        ]
    )
    return gauss / gauss.sum()


def _create_window(window_size: int, sigma: float, channel: int) -> torch.Tensor:
    """Creates a 2D Gaussian window.

    Args:
        window_size: The size of the window.
        sigma: The standard deviation of the Gaussian.
        channel: Number of input channels.

    Returns:
        A 4D Gaussian window tensor (channel, 1, window_size, window_size).
    """
    _1D_window = _gaussian(window_size, sigma).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window
