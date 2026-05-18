import torch
import torch.nn as nn
import torch.nn.functional as F


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


class CompositeLoss(nn.Module):
    """Combines Charbonnier and Wavelet domain constraints."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.charbonnier = CharbonnierLoss()
        self.wavelet = WaveletLoss()

    def forward(self, pred, target):
        l_char = self.charbonnier(pred, target)
        l_wave = self.wavelet(pred, target)

        # Total balanced loss using config weights
        total_loss = (
            self.config.charbonnier_weight * l_char
            + self.config.wavelet_weight * l_wave
        )
        return total_loss, {"loss_char": l_char, "loss_wave": l_wave}
