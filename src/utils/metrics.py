import torch
import torch.nn.functional as F
import numpy as np


def compute_psnr(pred, gt):
    """
    Computes PSNR on GPU for speed.
    Assumes tensors are in [0, 1] range.
    """
    mse = F.mse_loss(pred, gt)
    if mse == 0:
        return 100.0
    return 10 * torch.log10(1.0 / mse).item()


def compute_ssim(pred, gt, window_size=11, size_average=True):
    """
    Computes SSIM on GPU.
    Reference: https://github.com/Po-Hsun-Su/pytorch-ssim
    """
    # Ensure 4D
    if pred.dim() == 3:
        pred = pred.unsqueeze(0)
        gt = gt.unsqueeze(0)

    device = pred.device
    channel = pred.size(1)
    window = _create_window(window_size, channel).to(device)

    mu1 = F.conv2d(pred, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(gt, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(pred * pred, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(gt * gt, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(pred * gt, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean().item()
    else:
        return ssim_map.mean(1).mean(1).mean(1).item()


def _gaussian(window_size, sigma):
    gauss = torch.Tensor([np.exp(-(x - window_size // 2) ** 2 / float(2 * sigma**2)) for x in range(window_size)])
    return gauss / gauss.sum()


def _create_window(window_size, channel):
    _1D_window = _gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window
