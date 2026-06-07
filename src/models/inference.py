"""Advanced inference engine for HASST, supporting TTA and overlapping patch-based reconstruction."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from typing import Optional


class HASSTInferenceEngine:
    """Advanced inference engine implementing 8x Geometric Test-Time Augmentation (TTA).

    This engine supports TTA and seamless overlapping patch-based reconstruction
    for high-resolution images.
    """

    def __init__(self, model: nn.Module, device: Optional[torch.device] = None) -> None:
        """Initializes the inference engine.

        Args:
            model: The model to use for inference.
            device: The device to run inference on. If None, uses CUDA if available.
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(device)
        self.model.eval()
        self.device = device

    def _apply_tta(self, x: torch.Tensor, mode: int) -> torch.Tensor:
        """Applies one of the 8 geometric transformations for TTA.

        Args:
            x: Input tensor of shape (B, C, H, W).
            mode: Transformation mode (0-7).

        Returns:
            Transformed tensor.
        """
        if mode == 0:
            return x
        elif mode == 1:
            return torch.flip(x, dims=[2])  # Flip H
        elif mode == 2:
            return torch.flip(x, dims=[3])  # Flip W
        elif mode == 3:
            return torch.flip(x, dims=[2, 3])  # Flip HW
        elif mode == 4:
            return torch.rot90(x, k=1, dims=[2, 3])
        elif mode == 5:
            return torch.rot90(torch.flip(x, dims=[2]), k=1, dims=[2, 3])
        elif mode == 6:
            return torch.rot90(torch.flip(x, dims=[3]), k=1, dims=[2, 3])
        elif mode == 7:
            return torch.rot90(torch.flip(x, dims=[2, 3]), k=1, dims=[2, 3])
        return x

    def _invert_tta(self, x: torch.Tensor, mode: int) -> torch.Tensor:
        """Inverts the geometric transformation to realign with the source image.

        Args:
            x: Input tensor of shape (B, C, H, W).
            mode: Transformation mode (0-7) that was used to transform the image.

        Returns:
            Inverted tensor.
        """
        if mode == 0:
            return x
        elif mode == 1:
            return torch.flip(x, dims=[2])
        elif mode == 2:
            return torch.flip(x, dims=[3])
        elif mode == 3:
            return torch.flip(x, dims=[2, 3])
        elif mode == 4:
            return torch.rot90(
                x, k=3, dims=[2, 3]
            )  # Inverse of 90 deg rotation is 270 (k=3)
        elif mode == 5:
            return torch.flip(torch.rot90(x, k=3, dims=[2, 3]), dims=[2])
        elif mode == 6:
            return torch.flip(torch.rot90(x, k=3, dims=[2, 3]), dims=[3])
        elif mode == 7:
            return torch.flip(torch.rot90(x, k=3, dims=[2, 3]), dims=[2, 3])
        return x

    @torch.no_grad()
    def forward_tta(
        self, x: torch.Tensor, noise_prior: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Runs the 8x geometric self-ensemble forward pass.

        Args:
            x: Input tensor of shape (B, C, H, W).
            noise_prior: Optional noise prior tensor.

        Returns:
            Ensembled prediction.
        """
        x = x.to(self.device)
        tta_result = 0.0

        for mode in range(8):
            transformed_input = self._apply_tta(x, mode)

            # If a global noise_prior is provided, we must also apply the TTA transform to it
            # so that it aligns with the transformed input image.
            current_prior = None
            if noise_prior is not None:
                current_prior = self._apply_tta(noise_prior.to(self.device), mode)

            pred = self.model(transformed_input, noise_prior=current_prior)
            inverted_pred = self._invert_tta(pred, mode)
            # Accumulate in float32 to prevent precision drift from low-bit autocast
            tta_result = tta_result + (inverted_pred.float() / 8.0)

        return tta_result

    @torch.no_grad()
    def forward_tlc(
        self,
        x: torch.Tensor,
        patch_size: int = 256,
        noise_prior: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Test-Time Local Converter (TLC) inference wrapper.

        Splits high-resolution inputs into overlapping patches, infers each,
        and smoothly merges them using adaptive blending.

        Args:
            x: Input tensor of shape (B, C, H, W).
            patch_size: Size of the patches for inference.
            noise_prior: Optional noise prior tensor.

        Returns:
            Reconstructed prediction.
        """
        B, C, H, W = x.shape
        # Optimization: If the image matches patch_size exactly, skip overlapping logic
        if H == patch_size and W == patch_size:
            return self.forward_tta(x, noise_prior=noise_prior)

        return self.inference_patch_overlapping(
            x, patch_size=patch_size, stride=patch_size // 2, noise_prior=noise_prior
        )

    @torch.no_grad()
    def inference_patch_overlapping(
        self,
        x: torch.Tensor,
        patch_size: int = 256,
        stride: int = 192,
        noise_prior: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Inference using overlapping patches with smooth blending.

        Args:
            x: Input tensor of shape (B, C, H, W).
            patch_size: Size of the patches for inference.
            stride: Stride between patches.
            noise_prior: Optional noise prior tensor.

        Returns:
            Reconstructed prediction.
        """
        B, C, H, W = x.shape
        x = x.to(self.device)

        # To prevent window tapering from corrupting external boundaries, we pad the
        # image by the falloff amount, making actual image pixels well inside the map.
        falloff = max(1, patch_size // 8)
        pad_amount = falloff

        padded_x = F.pad(
            x, (pad_amount, pad_amount, pad_amount, pad_amount), mode="reflect"
        )

        # Pad noise_prior similarly if provided
        padded_prior = None
        if noise_prior is not None:
            padded_prior = F.pad(
                noise_prior.to(self.device),
                (pad_amount, pad_amount, pad_amount, pad_amount),
                mode="reflect",
            )

        # Now deal with dimensions not divisible by stride or patch size
        _, _, p_H, p_W = padded_x.shape
        pad_h_extra = (patch_size - p_H % patch_size) % patch_size
        pad_w_extra = (patch_size - p_W % patch_size) % patch_size

        if pad_h_extra > 0 or pad_w_extra > 0:
            padded_x = F.pad(padded_x, (0, pad_w_extra, 0, pad_h_extra), mode="reflect")
            if padded_prior is not None:
                padded_prior = F.pad(
                    padded_prior, (0, pad_w_extra, 0, pad_h_extra), mode="reflect"
                )

        _, _, new_H, new_W = padded_x.shape

        # Output and weight tracking canvases
        output_canvas = torch.zeros_like(padded_x)
        weight_canvas = torch.zeros((B, 1, new_H, new_W), device=self.device)

        # Create a smooth 2D windowing mask to soft-blend patch borders
        # We use a sine-based falloff which is smoother than linear at the transitions.
        dist = torch.arange(patch_size, device=self.device).float()
        # mask_1d will be 0 at edges and 1 in the middle
        mask_1d = torch.sin(
            torch.clamp(dist / (patch_size - 1) * torch.pi, 0, torch.pi)
        )
        # Square the mask to make the transition even smoother at the edges
        mask_1d = mask_1d**2
        window = mask_1d.reshape(1, 1, patch_size, 1) * mask_1d.reshape(
            1, 1, 1, patch_size
        )

        # Calculate range of patches ensuring the entire image (including padding) is covered
        y_range = list(range(0, new_H - patch_size + 1, stride))
        if not y_range or y_range[-1] != new_H - patch_size:
            y_range.append(new_H - patch_size)

        x_range = list(range(0, new_W - patch_size + 1, stride))
        if not x_range or x_range[-1] != new_W - patch_size:
            x_range.append(new_W - patch_size)

        total_patches = len(y_range) * len(x_range)

        # Slide over image grid
        pbar = tqdm(
            total=total_patches,
            desc="Overlapping patches",
            leave=False,
            disable=total_patches < 2,
        )
        for y in y_range:
            for x_coord in x_range:
                # Isolate crop
                patch = padded_x[
                    :, :, y : y + patch_size, x_coord : x_coord + patch_size
                ]

                patch_prior = None
                if padded_prior is not None:
                    patch_prior = padded_prior[
                        :, :, y : y + patch_size, x_coord : x_coord + patch_size
                    ]

                # Execute inference through the 8x TTA module
                pred_patch = self.forward_tta(patch, noise_prior=patch_prior)

                # Add to canvas using the window blending weight map
                output_canvas[
                    :, :, y : y + patch_size, x_coord : x_coord + patch_size
                ] += pred_patch * window
                weight_canvas[
                    :, :, y : y + patch_size, x_coord : x_coord + patch_size
                ] += window
                pbar.update(1)
        if total_patches >= 2:
            pbar.close()

        # Normalize across overlapped boundaries
        output_canvas /= torch.clamp(weight_canvas, min=1e-4)

        # Crop back down to original dimensions by skipping the initial falloff padding
        return output_canvas[
            :, :, pad_amount : pad_amount + H, pad_amount : pad_amount + W
        ]
