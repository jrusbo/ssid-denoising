import numpy as np
import torch
from typing import Optional, Tuple

# Global buffers for batch-size 1 augmentation
_ncm_noise_buffer: Optional[torch.Tensor] = None
_ncm_gt_buffer: Optional[torch.Tensor] = None
_afm_noisy_buffer: Optional[torch.Tensor] = None
_afm_gt_buffer: Optional[torch.Tensor] = None


def reset_augmentation_buffers() -> None:
    """Resets global temporal buffers used for batch-size 1 augmentations."""
    global _ncm_noise_buffer, _ncm_gt_buffer, _afm_noisy_buffer, _afm_gt_buffer
    _ncm_noise_buffer = None
    _ncm_gt_buffer = None
    _afm_noisy_buffer = None
    _afm_gt_buffer = None


def apply_noise_cutmix(
    noisy_batch: torch.Tensor, gt_batch: torch.Tensor, alpha: float = 0.2
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Applies NoiseCutMix augmentation.

    Extracts noise residuals and mixes them within the batch. Uses a temporal
    buffer if the batch size is 1 to enable cross-sample mixing.

    Args:
        noisy_batch: Tensor of noisy images [B, C, H, W].
        gt_batch: Tensor of ground truth images [B, C, H, W].
        alpha: Alpha parameter for Beta distribution.

    Returns:
        A tuple containing (mixed_noisy_batch, gt_batch).
    """
    global _ncm_noise_buffer, _ncm_gt_buffer
    B, C, H, W = noisy_batch.shape

    should_refresh_buffer = False
    if B > 1:
        # Mix within the batch
        noisy_b = torch.roll(noisy_batch, shifts=1, dims=0)
        gt_b = torch.roll(gt_batch, shifts=1, dims=0)
    else:
        # Use temporal buffer for BS=1
        if _ncm_noise_buffer is None or _ncm_noise_buffer.shape != noisy_batch.shape:
            _ncm_noise_buffer = noisy_batch.detach().clone()
            _ncm_gt_buffer = gt_batch.detach().clone()
            return noisy_batch, gt_batch

        noisy_b = _ncm_noise_buffer
        gt_b = _ncm_gt_buffer
        should_refresh_buffer = True

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

    if should_refresh_buffer:
        _ncm_noise_buffer.copy_(noisy_batch.detach())
        _ncm_gt_buffer.copy_(gt_batch.detach())

    return mixed_noisy, gt_batch


def adversarial_frequency_mixup(
    noisy_batch: torch.Tensor, gt_batch: torch.Tensor, alpha: float = 0.5
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Applies Adversarial Frequency Mixup (AFM).

    Mixes frequencies of both noisy and GT batches simultaneously.
    This ensures the mapping between noisy input and clean target remains valid.

    Args:
        noisy_batch: Tensor of noisy images [B, C, H, W].
        gt_batch: Tensor of ground truth images [B, C, H, W].
        alpha: Mixing ratio for frequencies.

    Returns:
        A tuple containing (mixed_noisy_batch, mixed_gt_batch).
    """
    global _afm_noisy_buffer, _afm_gt_buffer
    B = noisy_batch.shape[0]

    should_refresh_buffer = False
    if B > 1:
        noisy_b2 = torch.roll(noisy_batch, shifts=1, dims=0)
        gt_b2 = torch.roll(gt_batch, shifts=1, dims=0)
    else:
        if _afm_noisy_buffer is None or _afm_noisy_buffer.shape != noisy_batch.shape:
            _afm_noisy_buffer = noisy_batch.detach().clone()
            _afm_gt_buffer = gt_batch.detach().clone()
            return noisy_batch, gt_batch

        noisy_b2 = _afm_noisy_buffer
        gt_b2 = _afm_gt_buffer
        should_refresh_buffer = True

    def _mix_freq(img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        fft_1 = torch.fft.rfft2(img1, dim=(-2, -1), norm="ortho")
        fft_2 = torch.fft.rfft2(img2, dim=(-2, -1), norm="ortho")

        amp1, phase1 = torch.abs(fft_1), torch.angle(fft_1)
        amp2, _ = torch.abs(fft_2), torch.angle(fft_2)

        mixed_amp = (1 - alpha) * amp1 + alpha * amp2
        mixed_fft = torch.polar(mixed_amp, phase1)

        return torch.fft.irfft2(
            mixed_fft, s=img1.shape[-2:], dim=(-2, -1), norm="ortho"
        )

    mixed_noisy = torch.clamp(_mix_freq(noisy_batch, noisy_b2), 0.0, 1.0)
    mixed_gt = torch.clamp(_mix_freq(gt_batch, gt_b2), 0.0, 1.0)

    if should_refresh_buffer:
        _afm_noisy_buffer.copy_(noisy_batch.detach())
        _afm_gt_buffer.copy_(gt_batch.detach())

    return mixed_noisy, mixed_gt
