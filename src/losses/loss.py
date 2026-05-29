import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class CharbonnierLoss(nn.Module):
    """Robust L1 alternative that prevents unstable gradients near zero."""

    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps = eps * eps

    def forward(self, x, y):
        diff = x - y
        loss = torch.sqrt(diff * diff + self.eps)
        return loss.mean()


class WaveletLoss(nn.Module):
    """Penalizes errors in frequency sub-bands using Haar wavelets."""

    def __init__(self):
        super().__init__()
        # Define Haar wavelet filters manually for fast, differentiable execution
        h = torch.tensor([[0.5, 0.5], [0.5, 0.5]]).view(1, 1, 2, 2)
        lh = torch.tensor([[-0.5, -0.5], [0.5, 0.5]]).view(1, 1, 2, 2)
        hl = torch.tensor([[-0.5, 0.5], [-0.5, 0.5]]).view(1, 1, 2, 2)
        hh = torch.tensor([[0.5, -0.5], [-0.5, 0.5]]).view(1, 1, 2, 2)

        self.register_buffer("filters", torch.cat([h, lh, hl, hh], dim=0))

    def _dwt(self, x):
        B, C, H, W = x.shape
        # Filters are already on correct device via register_buffer and .to(device) in CompositeLoss
        x_unfolded = F.conv2d(x.reshape(B * C, 1, H, W), self.filters, stride=2, padding=0)
        return x_unfolded.reshape(B, C, 4, H // 2, W // 2)

    def forward(self, x, y):
        # Handle odd dimensions with padding if necessary
        pad_w = x.shape[-1] % 2
        pad_h = x.shape[-2] % 2
        if pad_w != 0 or pad_h != 0:
            x = F.pad(x, (0, pad_w, 0, pad_h))
            y = F.pad(y, (0, pad_w, 0, pad_h))
        return F.l1_loss(self._dwt(x), self._dwt(y))


class SSIMLoss(nn.Module):
    """Structural Similarity Index Measure (SSIM) Loss."""

    def __init__(self, window_size=11, sigma=1.5, channels=3, size_average=True):
        super().__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channels = channels
        self.window = self.create_window(window_size, channels)

    def gaussian(self, window_size, sigma):
        gauss = torch.Tensor([np.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
        return gauss / gauss.sum()

    def create_window(self, window_size, channel):
        _1D_window = self.gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window

    def forward(self, img1, img2):
        if img1.size(1) != self.channels:
            self.channels = img1.size(1)
            self.window = self.create_window(self.window_size, self.channels).to(img1.device)
        else:
            self.window = self.window.to(img1.device)

        mu1 = F.conv2d(img1, self.window, padding=self.window_size // 2, groups=self.channels)
        mu2 = F.conv2d(img2, self.window, padding=self.window_size // 2, groups=self.channels)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, self.window, padding=self.window_size // 2, groups=self.channels) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, self.window, padding=self.window_size // 2, groups=self.channels) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, self.window, padding=self.window_size // 2, groups=self.channels) - mu1_mu2

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        if self.size_average:
            return 1 - ssim_map.mean()
        else:
            return 1 - ssim_map.mean(1).mean(1).mean(1)


class CompositeLoss(nn.Module):
    """Combines Charbonnier, Wavelet and SSIM domain constraints."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.charbonnier = CharbonnierLoss()
        self.wavelet = WaveletLoss()
        self.ssim = SSIMLoss()

    def forward(self, pred, target):
        l_char = self.charbonnier(pred, target)
        l_wave = self.wavelet(pred, target)
        l_ssim = self.ssim(pred, target)

        # Total balanced loss using config weights
        total_loss = (
            self.config.charbonnier_weight * l_char
            + self.config.wavelet_weight * l_wave
            + self.config.ssim_weight * l_ssim
        )
        return total_loss, {"loss_char": l_char, "loss_wave": l_wave, "loss_ssim": l_ssim}
