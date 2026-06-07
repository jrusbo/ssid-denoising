import cv2
import numpy as np
import torch
import wandb
from pathlib import Path
from typing import Any, Dict, Optional, Union


class WandBValidationLogger:
    """Handles logging of metrics, gradients, and visual artifacts to Weights & Biases."""

    def __init__(
        self, config: Any, is_main_process: bool = True, run_id: Optional[str] = None
    ) -> None:
        """Initializes the WandBValidationLogger.

        Args:
            config: Configuration object containing project, entity, and logging settings.
            is_main_process: Whether this instance is running on the main process.
                Logging only occurs on the main process. Defaults to True.
            run_id: Optional ID to resume an existing WandB run.
        """
        self.is_main_process = is_main_process
        self.val_freq = config.val_freq
        self.run_id = run_id

        if self.is_main_process:
            # If run_id is provided, we resume that run. Otherwise, start new.
            wandb.init(
                project=config.wandb_project,
                entity=config.wandb_entity,
                config=dict(vars(config)),
                id=self.run_id,
                resume="allow",
            )
            self.run_id = wandb.run.id

    def get_run_id(self) -> Optional[str]:
        """Returns the current WandB run ID.

        Returns:
            The run ID as a string, or None if not initialized.
        """
        return self.run_id

    def log_metrics(
        self, step: int, metrics_dict: Dict[str, Any], commit: bool = True
    ) -> None:
        """Logs scalar metrics (Loss, PSNR, Learning Rate).

        Args:
            step: The current training step.
            metrics_dict: Dictionary of metric names and their values.
            commit: Whether to commit the log to WandB immediately. Defaults to True.
        """
        if not self.is_main_process:
            return
        wandb.log(metrics_dict, step=step, commit=commit)

    def log_gradients(
        self,
        step: int,
        model: torch.nn.Module,
        commit: bool = True,
        force_detailed: bool = False,
    ) -> None:
        """Logs gradient norms.

        Optimized to avoid multiple GPU-CPU syncs. Individual layer norms are only logged
        if force_detailed=True or at val_freq.

        Args:
            step: The current training step.
            model: The PyTorch model to log gradients for.
            commit: Whether to commit the log to WandB immediately. Defaults to True.
            force_detailed: If True, logs detailed per-layer gradient norms. Defaults to False.
        """
        if not self.is_main_process:
            return

        log_detailed = force_detailed or (step % self.val_freq == 0)

        names = []
        norms = []
        with torch.no_grad():
            for name, p in model.named_parameters():
                if p.grad is not None:
                    # Store the tensor norm (still on GPU)
                    names.append(name)
                    norms.append(p.grad.norm(2))

        if not norms:
            return

        # One single sync point: move all norms to CPU at once
        norms_cpu = torch.stack(norms).float().cpu().numpy()

        metrics = {}
        total_grad_norm_sq = 0.0

        for i, name in enumerate(names):
            norm_val = norms_cpu[i]
            total_grad_norm_sq += norm_val**2

            if log_detailed:
                if "weight" in name and ("conv" in name or "attn" in name):
                    short_name = name.replace("module.", "").replace("_orig_mod.", "")
                    metrics[f"grads/{short_name}"] = norm_val

        metrics["telemetry/total_gradient_norm"] = total_grad_norm_sq**0.5
        wandb.log(metrics, step=step, commit=commit)

    def log_visual_artifacts(
        self,
        step: int,
        noisy_tensor: torch.Tensor,
        pred_tensor: torch.Tensor,
        gt_tensor: torch.Tensor,
        noise_prior_tensor: Optional[torch.Tensor] = None,
        prefix: str = "visuals",
        commit: bool = True,
    ) -> None:
        """Stitches images into a single comparison grid and logs it to WandB.

        The grid layout is: [ Noisy | Prediction | Ground Truth | Error Map | Shot Prior | Read Prior ]
        where the last two panes are included if a noise prior is provided.

        Args:
            step: The current training step.
            noisy_tensor: The noisy input image tensor.
            pred_tensor: The predicted denoised image tensor.
            gt_tensor: The ground truth image tensor.
            noise_prior_tensor: Optional tensor containing noise priors. Defaults to None.
            prefix: Prefix for the logged artifact name. Defaults to "visuals".
            commit: Whether to commit the log to WandB immediately. Defaults to True.
        """
        if not self.is_main_process or (step % self.val_freq != 0):
            return

        def to_numpy(t: torch.Tensor) -> np.ndarray:
            if t.dim() == 4:
                t = t[0]
            # NumPy cannot ingest bfloat16 directly on some PyTorch builds.
            t = t.detach().float().cpu().clamp(0.0, 1.0)
            return (t.numpy().transpose(1, 2, 0) * 255).astype(np.uint8)

        def to_heatmap(t: torch.Tensor) -> np.ndarray:
            if t.dim() == 4:
                t = t[0]
            if t.dim() == 3:
                t = t[0]
            arr = t.detach().float().cpu().numpy()
            arr = arr - arr.min()
            denom = max(arr.max(), 1e-8)
            arr = (arr / denom * 255.0).astype(np.uint8)
            heat_bgr = cv2.applyColorMap(arr, cv2.COLORMAP_VIRIDIS)
            return cv2.cvtColor(heat_bgr, cv2.COLOR_BGR2RGB)

        noisy_img = to_numpy(noisy_tensor)
        pred_img = to_numpy(pred_tensor)
        gt_img = to_numpy(gt_tensor)

        # Calculate spatial error map and apply a colormap for better visibility
        error_map = np.abs(pred_img.astype(float) - gt_img.astype(float)).astype(
            np.uint8
        )
        error_map_gray = np.mean(error_map, axis=2).astype(np.uint8)
        error_map_color_bgr = cv2.applyColorMap(error_map_gray, cv2.COLORMAP_VIRIDIS)
        error_map_color_rgb = cv2.cvtColor(error_map_color_bgr, cv2.COLOR_BGR2RGB)

        panes = [noisy_img, pred_img, gt_img, error_map_color_rgb]

        if noise_prior_tensor is not None:
            if noise_prior_tensor.dim() == 4:
                noise_prior_tensor = noise_prior_tensor[0]
            if noise_prior_tensor.size(0) >= 2:
                panes.append(to_heatmap(noise_prior_tensor[0:1]))
                panes.append(to_heatmap(noise_prior_tensor[1:2]))

        comparison_grid = np.concatenate(panes, axis=1)

        wandb.log(
            {
                f"{prefix}/comparison_grid": wandb.Image(
                    comparison_grid,
                    caption="Left to Right: Noisy, HASST Prediction, Ground Truth, Error Map, Shot Prior, Read Prior",
                )
            },
            step=step,
            commit=commit,
        )

    def log_system_metrics(
        self, step: int, img_per_sec: float, gpu_mem_gb: float, commit: bool = True
    ) -> None:
        """Logs system performance and resource usage.

        Args:
            step: The current training step.
            img_per_sec: Throughput in images per second.
            gpu_mem_gb: GPU memory usage in GB.
            commit: Whether to commit the log to WandB immediately. Defaults to True.
        """
        if not self.is_main_process:
            return
        wandb.log(
            {
                "system/throughput_fps": img_per_sec,
                "system/gpu_mem_reserved_gb": gpu_mem_gb,
            },
            step=step,
            commit=commit,
        )

    def log_model_artifact(
        self,
        path: Union[str, Path],
        name: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Uploads a model checkpoint as a WandB Artifact.

        Args:
            path: Path to the checkpoint file.
            name: Base name for the artifact.
            metadata: Optional dictionary of metadata to associate with the artifact.
        """
        if not self.is_main_process or wandb.run is None:
            return

        abs_path = str(Path(path).resolve())

        # Append the run ID to the name to isolate artifacts per run.
        unique_name = f"{name}-{wandb.run.id}"
        artifact = wandb.Artifact(name=unique_name, type="model", metadata=metadata)
        artifact.add_file(abs_path)
        wandb.log_artifact(artifact)

    def finish(self) -> None:
        """Closes the WandB run."""
        if self.is_main_process and wandb.run is not None:
            wandb.finish()
