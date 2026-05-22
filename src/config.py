import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Union

import yaml


@dataclass
class Config:
    # --- Kaggle / Training Settings ---
    seed: int = 42
    max_hours: Optional[float] = 11.5
    use_compile: bool = True
    mixed_precision: str = "fp16"  # "no", "fp16", "bf16"

    # --- Paths ---
    data_dir: Union[str, Path] = "/kaggle/input/sidd-benchmark-srgb-psnr/Data"
    lmdb_dir: Union[str, Path] = "/kaggle/working/sidd_lmdb"
    output_dir: Union[str, Path] = "/kaggle/working/checkpoints"
    resume: bool = False
    resume_path: Optional[Union[str, Path]] = None

    # --- Model Hyperparameters ---
    embed_dim: int = 64
    num_blocks: int = 4
    in_channels: int = 3
    out_channels: int = 3

    # --- WandB & Checkpointing ---
    wandb_project: str = "kaggle-sidd-hasst"
    wandb_entity: Optional[str] = None
    wandb_run_id: Optional[str] = None
    log_freq: int = 100
    val_freq: int = 1000
    checkpoint_freq: int = 5000

    # --- Progressive Training Schedule ---
    # Phase 1: 128x128 (Fast iterations, global features), Phase 2: 256x256, Phase 3: 384x384 (Refinement)
    total_iters: int = 300000
    patch_sizes: List[int] = field(default_factory=lambda: [128, 256, 384])
    batch_sizes: List[int] = field(default_factory=lambda: [8, 4, 1])  # Per GPU
    phase_milestones: List[int] = field(default_factory=lambda: [100000, 200000])

    # --- Optimizer ---
    optimizer_type: str = "Adam"
    lr_initial: float = 1e-3
    lr_min: float = 1e-7
    beta1: float = 0.9
    beta2: float = 0.999
    weight_decay: float = 0.0

    # --- Loss Weights ---
    charbonnier_weight: float = 1.0
    wavelet_weight: float = 0.1

    # --- DDP / DataLoader Settings ---
    num_workers: int = 4
    pin_memory: bool = True

    def __post_init__(self):
        """Ensures that numeric types are correctly cast from YAML and paths are Path objects."""
        float_fields = [
            "max_hours", "lr_initial", "lr_min", "beta1", "beta2",
            "weight_decay", "charbonnier_weight", "wavelet_weight"
        ]
        int_fields = [
            "seed", "embed_dim", "num_blocks", "in_channels", "out_channels",
            "log_freq", "val_freq", "checkpoint_freq", "total_iters", "num_workers"
        ]
        
        for f in float_fields:
            val = getattr(self, f)
            if val is not None:
                if isinstance(val, str) and val.lower() in ["none", "null", ""]:
                    setattr(self, f, None)
                else:
                    setattr(self, f, float(val))
        
        for f in int_fields:
            val = getattr(self, f)
            if val is not None:
                if isinstance(val, str) and val.lower() in ["none", "null", ""]:
                    setattr(self, f, None)
                else:
                    setattr(self, f, int(val))
        
        # Convert string paths to Path objects
        self.data_dir = Path(self.data_dir)
        self.lmdb_dir = Path(self.lmdb_dir)
        self.output_dir = Path(self.output_dir)
        if self.resume_path:
            self.resume_path = Path(self.resume_path)

    @classmethod
    def load_from_yaml(cls, yaml_path: Union[str, Path]):
        with open(yaml_path, "r") as f:
            data = yaml.safe_load(f)
        return cls(**data)


def get_config():
    config_env = os.getenv("CONFIG_PATH")
    config_path = Path(config_env) if config_env else Path("configs/default.yaml")
    
    if config_path.exists():
        return Config.load_from_yaml(config_path)
    return Config()
