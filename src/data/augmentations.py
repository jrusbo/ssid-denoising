import random

import numpy as np
import torch

# Global buffers for batch-size 1 augmentation
_noise_buffer = None
_gt_buffer = None
_afm_buffer = None

def reset_augmentation_buffers():
    """Resets global temporal buffers used for BS=1 augmentations."""
    global _noise_buffer, _gt_buffer, _afm_buffer
    _noise_buffer = None
    _gt_buffer = None
    _afm_buffer = None

def apply_noise_cutmix(noisy_batch, gt_batch, alpha=0.2):
    """
    Batched NoiseCutMix: Extracts noise residuals and mixes them within the batch.
    Uses a temporal buffer if batch size is 1.
    """
    global _noise_buffer, _gt_buffer
    B, C, H, W = noisy_batch.shape
    
    if B > 1:
        # Mix within the batch
        noisy_b = torch.roll(noisy_batch, shifts=1, dims=0)
        gt_b = torch.roll(gt_batch, shifts=1, dims=0)
    else:
        # Use temporal buffer for BS=1
        if _noise_buffer is None or _noise_buffer.shape != noisy_batch.shape:
            _noise_buffer = noisy_batch.detach().clone()
            _gt_buffer = gt_batch.detach().clone()
            return noisy_batch, gt_batch
        
        noisy_b = _noise_buffer
        gt_b = _gt_buffer
        # Update buffer for next iteration
        _noise_buffer = noisy_batch.detach().clone()
        _gt_buffer = gt_batch.detach().clone()

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
    noise_res_a = noisy_batch - gt_batch
    noise_res_b = noisy_b - gt_b

    # Mix noise residuals
    mixed_noise = noise_res_a.clone()
    mixed_noise[:, :, y1:y2, x1:x2] = noise_res_b[:, :, y1:y2, x1:x2]

    # Apply mixed noise to the baseline (gt_batch)
    mixed_noisy = gt_batch + mixed_noise

    return mixed_noisy, gt_batch


def adversarial_frequency_mixup(batch, alpha=0.5):
    """
    Batched AFM: Mixes frequencies within the batch on GPU.
    Uses a temporal buffer if batch size is 1.
    """
    global _afm_buffer
    B = batch.shape[0]
    
    if B > 1:
        batch2 = torch.roll(batch, shifts=1, dims=0)
    else:
        if _afm_buffer is None or _afm_buffer.shape != batch.shape:
            _afm_buffer = batch.detach().clone()
            return batch
        
        batch2 = _afm_buffer
        _afm_buffer = batch.detach().clone()

    # Apply 2D FFT along spatial dimensions
    fft_1 = torch.fft.rfft2(batch, dim=(-2, -1), norm="ortho")
    fft_2 = torch.fft.rfft2(batch2, dim=(-2, -1), norm="ortho")

    # Linear mix of the amplitude spectrum while keeping phase
    amp1, phase1 = torch.abs(fft_1), torch.angle(fft_1)
    amp2, _ = torch.abs(fft_2), torch.angle(fft_2)

    mixed_amp = (1 - alpha) * amp1 + alpha * amp2
    mixed_fft = torch.polar(mixed_amp, phase1)

    # Inverse FFT to get back to spatial domain
    mixed_img = torch.fft.irfft2(
        mixed_fft, s=batch.shape[-2:], dim=(-2, -1), norm="ortho"
    )
    return torch.clamp(mixed_img, 0.0, 1.0)
