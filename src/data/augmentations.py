import random

import numpy as np
import torch


def apply_noise_cutmix(noisy_a, gt_a, noisy_b, gt_b, alpha=0.2):
    """
    NoiseCutMix: Extracts the noise residuals (Difference between Noisy and Ground Truth)
    from two distinct image patches and mixes them before applying them to a structural baseline.
    """
    if random.random() > 0.5:
        return noisy_a, gt_a

    B, C, H, W = noisy_a.shape if len(noisy_a.shape) == 4 else (1, *noisy_a.shape)
    if len(noisy_a.shape) == 3:
        noisy_a = noisy_a.unsqueeze(0)
        gt_a = gt_a.unsqueeze(0)
        noisy_b = noisy_b.unsqueeze(0)
        gt_b = gt_b.unsqueeze(0)

    lam = np.random.beta(alpha, alpha)

    cx = np.random.randint(W)
    cy = np.random.randint(H)
    cut_w = int(W * np.sqrt(1.0 - lam))
    cut_h = int(H * np.sqrt(1.0 - lam))

    x1 = np.clip(cx - cut_w // 2, 0, W)
    y1 = np.clip(cy - cut_h // 2, 0, H)
    x2 = np.clip(cx + cut_w // 2, 0, W)
    y2 = np.clip(cy + cut_h // 2, 0, H)

    # Extract noise residuals
    noise_res_a = noisy_a - gt_a
    noise_res_b = noisy_b - gt_b

    # Structural baseline: we can use gt_a as the baseline.
    # We mix the noise residuals according to the cutmix mask.
    mixed_noise = noise_res_a.clone()
    mixed_noise[:, :, y1:y2, x1:x2] = noise_res_b[:, :, y1:y2, x1:x2]

    # Apply mixed noise to the baseline (gt_a)
    mixed_noisy = gt_a + mixed_noise

    # Both images have the same ground truth structure now (gt_a)
    mixed_gt = gt_a

    if B == 1:
        return mixed_noisy.squeeze(0), mixed_gt.squeeze(0)

    return mixed_noisy, mixed_gt


def adversarial_frequency_mixup(img1, img2, alpha=0.5):
    """
    AFM: Mixes the low/high frequency components of two images in the Fourier Domain
    to generalize across different illumination styles and camera ISPs.
    """
    if random.random() > 0.5:
        return img1

    # Apply 2D FFT along spatial dimensions
    fft_1 = torch.fft.rfft2(img1, dim=(-2, -1), norm="ortho")
    fft_2 = torch.fft.rfft2(img2, dim=(-2, -1), norm="ortho")

    # Linear mix of the amplitude spectrum while keeping phase
    amp1, phase1 = torch.abs(fft_1), torch.angle(fft_1)
    amp2, _ = torch.abs(fft_2), torch.angle(fft_2)

    mixed_amp = (1 - alpha) * amp1 + alpha * amp2
    mixed_fft = torch.polar(mixed_amp, phase1)

    # Inverse FFT to get back to spatial domain
    mixed_img = torch.fft.irfft2(
        mixed_fft, s=img1.shape[-2:], dim=(-2, -1), norm="ortho"
    )
    return torch.clamp(mixed_img, 0.0, 1.0)


def simple_isp_inversion(srgb_tensor):
    """
    Approximates unprocessing by reversing gamma correction to bring sRGB
    back into a pseudo-linear RAW space for realistic noise injection.
    """
    # Inverse Gamma correction: approximation of sRGB -> Linear
    linear_raw = torch.pow(srgb_tensor, 2.2)
    return linear_raw


def simple_isp_forward(linear_raw):
    """Re-applies gamma correction to bring linear RAW back to sRGB."""
    return torch.pow(torch.clamp(linear_raw, 1e-6, 1.0), 1.0 / 2.2)
