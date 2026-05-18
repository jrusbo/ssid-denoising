import cv2
import numpy as np
import wandb
from pathlib import Path


class WandBValidationLogger:
    def __init__(self, config, is_main_process=True, run_id=None):
        self.is_main_process = is_main_process
        self.val_freq = config.val_freq
        self.run_id = run_id

        if self.is_main_process:
            # If run_id is provided, we resume that run. Otherwise, start new.
            wandb.init(
                project=config.wandb_project,
                entity=config.wandb_entity,
                config=vars(config),
                id=self.run_id,
                resume="allow"
            )
            # Store the run_id (either the one provided or the new one generated)
            self.run_id = wandb.run.id

    def get_run_id(self):
        return self.run_id

    def log_metrics(self, step, metrics_dict, commit=True):
        """Logs scalar metrics (Loss, PSNR, Learning Rate)."""
        if not self.is_main_process:
            return
        wandb.log(metrics_dict, step=step, commit=commit)

    def log_gradients(self, step, model, commit=True):
        """Logs gradient norms to spot gradient explosions or vanishing layers early."""
        if not self.is_main_process:
            return

        metrics = {}
        total_grad_norm = 0.0
        
        # Log global norm and check for specific modules if possible
        for name, p in model.named_parameters():
            if p.grad is not None:
                param_norm = p.grad.norm(2).item()
                total_grad_norm += param_norm ** 2
                
                # Log norms for major components (e.g., specific blocks or layers)
                # but limit to avoid overwhelming wandb
                if "weight" in name and ("conv" in name or "attn" in name):
                    # Shorten name for readability in wandb
                    short_name = name.replace("module.", "").replace("_orig_mod.", "")
                    metrics[f"grads/{short_name}"] = param_norm

        metrics["telemetry/total_gradient_norm"] = total_grad_norm**0.5
        wandb.log(metrics, step=step, commit=commit)

    def log_visual_artifacts(self, step, noisy_tensor, pred_tensor, gt_tensor, prefix="visuals", commit=True):
        """
        Stitches images into a single comparison grid:
        [ Noisy | Prediction | Ground Truth | Error Map ]
        """
        if not self.is_main_process or (step % self.val_freq != 0):
            return

        # Isolate the first example out of the batch
        def to_numpy(t):
            # Handle both (B, C, H, W) and (C, H, W)
            if t.dim() == 4:
                t = t[0]
            return (t.detach().cpu().clamp(0.0, 1.0).numpy().transpose(1, 2, 0) * 255).astype(np.uint8)

        # Convert tensors to numpy images (RGB)
        noisy_img = to_numpy(noisy_tensor)
        pred_img = to_numpy(pred_tensor)
        gt_img = to_numpy(gt_tensor)
        
        # Calculate spatial error map
        error_map = np.abs(pred_img.astype(float) - gt_img.astype(float)).astype(np.uint8)
        # Average across channels for a single intensity map
        error_map_gray = np.mean(error_map, axis=2).astype(np.uint8)
        
        # Apply a colormap to the error map for better visibility
        # OpenCV applyColorMap expects BGR
        error_map_color_bgr = cv2.applyColorMap(error_map_gray, cv2.COLORMAP_VIRIDIS)
        # Convert BGR (OpenCV) back to RGB
        error_map_color_rgb = cv2.cvtColor(error_map_color_bgr, cv2.COLOR_BGR2RGB)

        # Stitch horizontally
        comparison_grid = np.concatenate([noisy_img, pred_img, gt_img, error_map_color_rgb], axis=1)

        wandb.log(
            {
                f"{prefix}/comparison_grid": wandb.Image(
                    comparison_grid, 
                    caption="Left to Right: Noisy, HASST Prediction, Ground Truth, Error Map (Viridis)"
                )
            },
            step=step,
            commit=commit
        )

    def log_system_metrics(self, step, img_per_sec, gpu_mem_gb, commit=True):
        """Logs system performance and resource usage."""
        if not self.is_main_process:
            return
        wandb.log({
            "system/throughput_fps": img_per_sec,
            "system/gpu_mem_reserved_gb": gpu_mem_gb
        }, step=step, commit=commit)

    def log_model_artifact(self, path, name, metadata=None):
        """Uploads a model checkpoint as a WandB Artifact, unique to this run ID."""
        if not self.is_main_process or wandb.run is None:
            return
            
        # Ensure we use the absolute path for the artifact file
        abs_path = str(Path(path).resolve())
        
        # Append the run ID to the name to isolate artifacts per run.
        # This prevents unrelated runs from cluttering each other's version history.
        unique_name = f"{name}-{wandb.run.id}"
        artifact = wandb.Artifact(name=unique_name, type="model", metadata=metadata)
        artifact.add_file(abs_path)
        wandb.log_artifact(artifact)

    def finish(self):
        """Closes the WandB run."""
        if self.is_main_process and wandb.run is not None:
            wandb.finish()
