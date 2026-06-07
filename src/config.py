import os
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import List, Optional, Union

import yaml


@dataclass
class Config:
    """Configuration class for SIDD denoising training and inference.

    Attributes:
        seed: Random seed for reproducibility.
        max_hours: Maximum training time in hours.
        use_compile: Whether to use torch.compile for speed.
        mixed_precision: Precision mode ('no', 'fp16', 'bf16').
        lmdb_dir: Path to the LMDB database directory.
        output_dir: Path to the directory where checkpoints and logs are saved.
        resume: Whether to resume training from a checkpoint.
        resume_path: Specific path to a checkpoint to resume from.
        embed_dim: Base embedding dimension for the model.
        num_blocks: Total number of blocks in the model.
        in_channels: Number of input image channels.
        out_channels: Number of output image channels.
        lonpe_scale_physical: Whether to use physical scaling for LoNPE.
        lonpe_shot_range: Range for shot noise parameters.
        lonpe_read_range: Range for read noise parameters.
        wandb_project: Weights & Biases project name.
        wandb_entity: Weights & Biases entity (username or team).
        wandb_run_id: Specific WandB run ID to resume.
        log_freq: Frequency of logging training metrics.
        val_freq: Frequency of validation runs.
        checkpoint_freq: Frequency of saving checkpoints.
        wandb_watch: Whether to use wandb.watch to log gradients.
        wandb_watch_log_freq: Frequency of wandb.watch logging.
        require_mamba: If True, raises an error if 'mamba-ssm' is missing.
        total_iters: Total number of training iterations.
        patch_sizes: List of patch sizes for progressive training phases.
        batch_sizes: List of batch sizes corresponding to patch sizes.
        phase_milestones: Iteration counts where phase transitions occur.
        lr_initial: Initial learning rate.
        lr_min: Minimum learning rate for cosine annealing.
        beta1: Beta1 parameter for AdamW optimizer.
        beta2: Beta2 parameter for AdamW optimizer.
        weight_decay: Weight decay for AdamW optimizer.
        charbonnier_weight: Weight for Charbonnier loss.
        wavelet_weight: Weight for Wavelet loss.
        ssim_weight: Weight for SSIM loss.
        num_workers: Number of DataLoader workers.
        pin_memory: Whether to pin memory in DataLoader.
        prefetch_factor: Prefetch factor for DataLoader.
        train_loader_in_order: Whether to maintain order in training loader.
        worker_cpu_threads: Number of CPU threads per worker.
        channels_last: Whether to use channels_last memory format.
        cudnn_benchmark: Whether to enable cuDNN benchmarking.
        gradient_accumulation_steps: Number of steps for gradient accumulation.
    """

    # --- Kaggle / Training Settings ---
    seed: int = 42
    max_hours: Optional[float] = 11.5
    use_compile: bool = True
    mixed_precision: str = "fp16"

    # --- Paths ---
    lmdb_dir: Union[str, Path] = "/kaggle/working/sidd_lmdb"
    output_dir: Union[str, Path] = "/kaggle/working/checkpoints"
    resume: bool = False
    resume_path: Optional[Union[str, Path]] = None

    # --- Model ---
    embed_dim: int = 37
    num_blocks: int = 20
    in_channels: int = 3
    out_channels: int = 3
    lonpe_scale_physical: bool = True
    lonpe_shot_range: List[float] = field(default_factory=lambda: [1.0e-5, 5.0e-1])
    lonpe_read_range: List[float] = field(default_factory=lambda: [1.0e-6, 1.0e-2])

    # --- WandB & Checkpointing ---
    wandb_project: str = "kaggle-sidd-hasst"
    wandb_entity: Optional[str] = None
    wandb_run_id: Optional[str] = None
    log_freq: int = 100
    val_freq: int = 1000
    checkpoint_freq: int = 5000
    wandb_watch: bool = True
    wandb_watch_log_freq: int = 100
    require_mamba: bool = False

    # --- Progressive Training Schedule ---
    total_iters: int = 300000
    patch_sizes: List[int] = field(default_factory=lambda: [128, 160, 192, 256, 384])
    batch_sizes: List[int] = field(default_factory=lambda: [8, 5, 4, 2, 1])
    phase_milestones: List[int] = field(
        default_factory=lambda: [60000, 120000, 180000, 240000]
    )

    # --- Optimizer ---
    lr_initial: float = 1.0e-4
    lr_min: float = 1.0e-7
    beta1: float = 0.9
    beta2: float = 0.999
    weight_decay: float = 0.0

    # --- Loss Weights ---
    charbonnier_weight: float = 1.0
    wavelet_weight: float = 0.05
    ssim_weight: float = 0.0

    # --- DDP / DataLoader Settings ---
    num_workers: int = 4
    pin_memory: bool = True
    prefetch_factor: int = 4
    train_loader_in_order: bool = False
    worker_cpu_threads: int = 1

    # --- Performance Optimizations ---
    channels_last: bool = True
    cudnn_benchmark: bool = True
    gradient_accumulation_steps: int = 1

    def __post_init__(self) -> None:
        """Ensures that numeric types are correctly cast from YAML and paths are Path objects."""
        for f in fields(self):
            val = getattr(self, f.name)
            if val is None:
                continue

            # Handle string "None" or "null" from YAML
            if isinstance(val, str) and val.lower() in ["none", "null", ""]:
                setattr(self, f.name, None)
                continue

            # Automatic casting based on type hint
            if f.type is int or f.type == Optional[int]:
                setattr(self, f.name, int(val))
            elif f.type is float or f.type == Optional[float]:
                setattr(self, f.name, float(val))
            elif (
                "Path" in str(f.type)
                or f.name.endswith("_dir")
                or f.name.endswith("_path")
            ):
                setattr(self, f.name, Path(val))

        # Explicitly ensure these are Path objects
        self.lmdb_dir = Path(self.lmdb_dir)
        self.output_dir = Path(self.output_dir)
        if self.resume_path:
            self.resume_path = Path(self.resume_path)

    @classmethod
    def load_from_yaml(cls, yaml_path: Union[str, Path]) -> "Config":
        """Loads configuration from a YAML file.

        Args:
            yaml_path: Path to the YAML configuration file.

        Returns:
            A Config instance with values from the YAML file.
        """
        with open(yaml_path, "r") as f:
            data = yaml.safe_load(f)
        if data is None:
            return cls()
        return cls(**data)


def get_config() -> Config:
    """Retrieves the configuration based on environment variables or default paths.

    Returns:
        The Config instance to be used for the current run.
    """
    config_env = os.getenv("CONFIG_PATH")
    if config_env:
        config_path = Path(config_env)
        if config_path.exists():
            return Config.load_from_yaml(config_path)

    # Check for default config in the configs directory
    default_config = Path("configs/default.yaml")
    if default_config.exists():
        return Config.load_from_yaml(default_config)

    # If no config found, return default Config instance
    return Config()
