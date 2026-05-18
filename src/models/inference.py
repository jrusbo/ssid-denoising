import torch
import torch.nn.functional as F
from tqdm import tqdm


class HASSTInferenceEngine:
    """
    Advanced inference engine implementing 8x Geometric Test-Time Augmentation (TTA)
    and seamless overlapping patch-based reconstruction for high-resolution images.
    """

    def __init__(self, model, device=None):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(device)
        self.model.eval()
        self.device = device

    def _apply_tta(self, x: torch.Tensor, mode: int) -> torch.Tensor:
        """Applies one of the 8 geometric transformations for TTA."""
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
        """Inverts the geometric transformation to realign with the source image."""
        if mode == 0:
            return x
        elif mode == 1:
            return torch.flip(x, dims=[2])
        elif mode == 2:
            return torch.flip(x, dims=[3])
        elif mode == 3:
            return torch.flip(x, dims=[2, 3])
        elif mode == 4:
            return torch.rot90(x, k=3, dims=[2, 3])  # Inverse of 90 deg rotation is 270 (k=3)
        elif mode == 5:
            return torch.flip(torch.rot90(x, k=3, dims=[2, 3]), dims=[2])
        elif mode == 6:
            return torch.flip(torch.rot90(x, k=3, dims=[2, 3]), dims=[3])
        elif mode == 7:
            return torch.flip(torch.rot90(x, k=3, dims=[2, 3]), dims=[2, 3])
        return x

    @torch.no_grad()
    def forward_tta(self, x: torch.Tensor) -> torch.Tensor:
        """Runs the 8x geometric self-ensemble forward pass with running average."""
        x = x.to(self.device)
        tta_result = 0.0

        for mode in tqdm(range(8), desc="TTA forward pass", leave=False):
            transformed_input = self._apply_tta(x, mode)
            pred = self.model(transformed_input)
            inverted_pred = self._invert_tta(pred, mode)
            tta_result = tta_result + (inverted_pred / 8.0)

        return tta_result

    @torch.no_grad()
    def inference_patch_overlapping(
        self, x: torch.Tensor, patch_size=256, stride=192
    ) -> torch.Tensor:
        """
        Splits high-resolution validation images into overlapping windows,
        runs TTA inference, and blends boundaries seamlessly using a weight matrix.
        """
        B, C, H, W = x.shape
        x = x.to(self.device)

        # Output and weight tracking canvases
        output_canvas = torch.zeros_like(x)
        weight_canvas = torch.zeros((B, 1, H, W), device=self.device)

        # Create a linear 2D windowing mask to soft-blend patch borders
        # Vectorized for speed and efficiency
        dist = torch.arange(patch_size, device=self.device)
        dist = torch.minimum(dist, patch_size - 1 - dist).float()
        falloff = max(1, patch_size // 8)
        mask_1d = (dist / falloff).clamp(0.0, 1.0)
        window = (mask_1d.reshape(1, 1, patch_size, 1) * mask_1d.reshape(1, 1, 1, patch_size))

        # Pad image to handle dimensions not cleanly divisible by patch configurations
        pad_h = (patch_size - H % patch_size) % patch_size
        pad_w = (patch_size - W % patch_size) % patch_size
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
            output_canvas = F.pad(output_canvas, (0, pad_w, 0, pad_h))
            weight_canvas = F.pad(weight_canvas, (0, pad_w, 0, pad_h))

        _, _, new_H, new_W = x.shape

        # Calculate range of patches ensuring the entire image (including padding) is covered
        y_range = list(range(0, new_H - patch_size + 1, stride))
        if not y_range or y_range[-1] != new_H - patch_size:
            y_range.append(new_H - patch_size)
            
        x_range = list(range(0, new_W - patch_size + 1, stride))
        if not x_range or x_range[-1] != new_W - patch_size:
            x_range.append(new_W - patch_size)

        total_patches = len(y_range) * len(x_range)

        # Slide over image grid
        pbar = tqdm(total=total_patches, desc="Overlapping patches", leave=False, disable=total_patches < 2)
        for y in y_range:
            for x_coord in x_range:
                # Isolate crop
                patch = x[:, :, y : y + patch_size, x_coord : x_coord + patch_size]

                # Execute inference through the 8x TTA module
                pred_patch = self.forward_tta(patch)

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

        # Crop back down to original dimensions if padded earlier
        return output_canvas[:, :, :H, :W]
