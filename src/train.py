import argparse
import inspect
import json
import logging
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.optim as optim
import wandb
from accelerate import Accelerator
from torch.utils.data import DataLoader
from tqdm import tqdm

# Import our custom modules
from config import Config
from data.dataset import SIDDDatasetLMDB
from data.augmentations import (
    adversarial_frequency_mixup,
    apply_noise_cutmix,
    reset_augmentation_buffers,
)
from losses.loss import CompositeLoss
from models.hasst import HASST
from utils.logger import WandBValidationLogger
from utils.metrics import compute_psnr, compute_ssim


# Setup logging for multi-config runs
logging.basicConfig(level=logging.INFO)
logger_cli = logging.getLogger(__name__)


def make_worker_init_fn(
    base_seed: int, worker_cpu_threads: int
) -> Callable[[int], None]:
    """Create deterministic per-worker setup without CPU thread oversubscription.

    Args:
        base_seed: Base seed for random number generation.
        worker_cpu_threads: Number of CPU threads to use per worker.

    Returns:
        A worker initialization function.
    """

    def _worker_init_fn(worker_id: int) -> None:
        worker_seed = base_seed + worker_id
        random.seed(worker_seed)
        np.random.seed(worker_seed % (2**32 - 1))
        torch.manual_seed(worker_seed)

        if worker_cpu_threads and worker_cpu_threads > 0:
            torch.set_num_threads(worker_cpu_threads)
            try:
                torch.set_num_interop_threads(1)
            except RuntimeError:
                pass

    return _worker_init_fn


def get_raw_model(model: torch.nn.Module, accelerator: Accelerator) -> torch.nn.Module:
    """Unwraps model from Accelerator and torch.compile for saving.

    Args:
        model: The model to unwrap.
        accelerator: The Accelerator instance.

    Returns:
        The unwrapped base nn.Module.
    """
    while hasattr(model, "module") or hasattr(model, "_orig_mod"):
        if hasattr(model, "module"):
            model = model.module
        if hasattr(model, "_orig_mod"):
            model = model._orig_mod

    return accelerator.unwrap_model(model)


@torch.no_grad()
def evaluate_pipeline(
    model: torch.nn.Module, dataloader: DataLoader, accelerator: Accelerator
) -> Tuple[
    float,
    float,
    Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]],
]:
    """Evaluates the model on validation sample sets.

    Args:
        model: The model to evaluate.
        dataloader: Validation DataLoader.
        accelerator: The Accelerator instance.

    Returns:
        A tuple of (average_psnr, average_ssim, visualization_sample).
    """
    model.eval()
    local_psnr = torch.tensor(0.0, device=accelerator.device)
    local_ssim = torch.tensor(0.0, device=accelerator.device)
    local_samples = torch.tensor(0.0, device=accelerator.device)
    val_sample = None
    raw_model = get_raw_model(model, accelerator)

    pbar = tqdm(
        dataloader,
        desc="Evaluating",
        disable=not accelerator.is_main_process,
        leave=False,
        mininterval=2.0,
    )

    with accelerator.autocast():
        for i, (noisy, gt) in enumerate(pbar):
            pred = model(noisy)

            if i == 0 and accelerator.is_main_process:
                noise_prior = None
                if hasattr(raw_model, "estimate_noise_prior"):
                    noise_prior = raw_model.estimate_noise_prior(noisy).detach()
                val_sample = (noisy.detach(), pred.detach(), gt.detach(), noise_prior)

            local_psnr += compute_psnr(pred, gt)
            local_ssim += compute_ssim(pred, gt, size_average=False)
            local_samples += pred.shape[0]

    metrics = torch.stack([local_psnr, local_ssim, local_samples])

    if accelerator.num_processes > 1:
        metrics = accelerator.reduce(metrics, reduction="sum")

    global_psnr, global_ssim, total_samples = metrics.tolist()

    model.train()
    if total_samples == 0:
        return 0.0, 0.0, None
    return global_psnr / total_samples, global_ssim / total_samples, val_sample


def create_dataloaders(
    cfg: Config, patch_size: int, batch_size: int
) -> Tuple[DataLoader, DataLoader, float]:
    """Dynamically creates dataloaders for the progressive learning schedule.

    Args:
        cfg: The configuration instance.
        patch_size: Patch size for this phase.
        batch_size: Batch size for this phase.

    Returns:
        A tuple of (train_loader, val_loader, initialization_time).
    """
    start_time = time.time()
    train_dataset = SIDDDatasetLMDB(
        lmdb_dir=cfg.lmdb_dir, patch_size=patch_size, split="train", seed=cfg.seed
    )
    val_dataset = SIDDDatasetLMDB(
        lmdb_dir=cfg.lmdb_dir, patch_size=patch_size, split="val", seed=cfg.seed
    )
    dataset_time = time.time() - start_time
    dataloader_signature = inspect.signature(DataLoader).parameters

    train_loader_kwargs: Dict[str, Any] = dict(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        drop_last=True,
        persistent_workers=(cfg.num_workers > 0),
    )
    if cfg.num_workers > 0:
        train_loader_kwargs["prefetch_factor"] = cfg.prefetch_factor
        train_loader_kwargs["worker_init_fn"] = make_worker_init_fn(
            cfg.seed,
            cfg.worker_cpu_threads,
        )

    if "in_order" in dataloader_signature:
        train_loader_kwargs["in_order"] = cfg.train_loader_in_order

    if (
        "pin_memory_device" in dataloader_signature
        and cfg.pin_memory
        and torch.cuda.is_available()
    ):
        train_loader_kwargs["pin_memory_device"] = f"cuda:{torch.cuda.current_device()}"

    train_loader = DataLoader(**train_loader_kwargs)

    val_workers = min(2, cfg.num_workers)
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=val_workers,
        pin_memory=cfg.pin_memory,
        persistent_workers=False,
    )

    return train_loader, val_loader, dataset_time


def run_training(cfg: Config, start_time: Optional[float] = None) -> None:
    """Main training loop for a single configuration.

    Args:
        cfg: The configuration instance.
        start_time: Global start time for time-limited runs.
    """
    init_start = time.time()
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    reset_augmentation_buffers()

    accelerator = Accelerator(
        mixed_precision=cfg.mixed_precision,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
    )

    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = cfg.cudnn_benchmark

    if accelerator.is_main_process:
        accelerator.print(f"Accelerator initialized in {time.time() - init_start:.2f}s")
        accelerator.print("\n" + "=" * 50)
        accelerator.print("RECEIVED CONFIGURATION:")
        cfg_dict = asdict(cfg)
        for k, v in cfg_dict.items():
            if isinstance(v, (str, Path)) and any(x in k for x in ["dir", "path"]):
                try:
                    cfg_dict[k] = str(Path(v).resolve())
                except Exception:
                    pass
        accelerator.print(json.dumps(cfg_dict, indent=4))
        accelerator.print("=" * 50 + "\n")

    if start_time is None:
        start_time = time.time()
    max_seconds = cfg.max_hours * 3600 if cfg.max_hours and cfg.max_hours > 0 else None

    global_step = 0
    current_phase = 0
    best_psnr = 0.0
    wandb_run_id = None
    rng_states = None
    checkpoint_data = None

    base_output_dir = Path(cfg.output_dir).resolve()
    base_output_dir.mkdir(parents=True, exist_ok=True)

    if cfg.resume:
        ckpt_load_start = time.time()
        ckpt_path = None
        if cfg.resume_path and Path(cfg.resume_path).exists():
            ckpt_path = Path(cfg.resume_path).resolve()
        else:
            last_models = list(base_output_dir.glob("*/last_model.pth"))
            if last_models:
                ckpt_path = sorted(last_models, key=lambda x: x.stat().st_mtime)[-1]

        if ckpt_path:
            if ckpt_path.is_dir():
                ckpt_path = ckpt_path / "last_model.pth"

            if not ckpt_path.exists():
                accelerator.print(f"Warning: Checkpoint file {ckpt_path} not found.")
                ckpt_path = None
            else:
                accelerator.print(f"Resuming from {ckpt_path}...")
                try:
                    checkpoint_data = torch.load(
                        ckpt_path, map_location="cpu", weights_only=False
                    )
                except TypeError:
                    checkpoint_data = torch.load(ckpt_path, map_location="cpu")

            if ckpt_path:
                accelerator.print(
                    f"Checkpoint loaded in {time.time() - ckpt_load_start:.2f}s"
                )
                global_step = checkpoint_data["global_step"]
                current_phase = checkpoint_data.get("current_phase", 0)
                best_psnr = checkpoint_data.get("best_psnr", 0.0)

                wandb_run_id = checkpoint_data.get("wandb_run_id") or cfg.wandb_run_id
                rng_states = checkpoint_data.get("rng_states")

                if wandb_run_id:
                    accelerator.print(f"Found WandB Run ID to resume: {wandb_run_id}")

                while (
                    current_phase < len(cfg.phase_milestones)
                    and global_step >= cfg.phase_milestones[current_phase]
                ):
                    current_phase += 1

                current_phase = min(
                    current_phase, len(cfg.patch_sizes) - 1, len(cfg.batch_sizes) - 1
                )
        else:
            accelerator.print(
                "Warning: 'resume' is True but no valid checkpoint was found."
            )
            wandb_run_id = cfg.wandb_run_id

    logger_init_start = time.time()
    logger = WandBValidationLogger(
        cfg, is_main_process=accelerator.is_main_process, run_id=wandb_run_id
    )

    run_info = [None, None]
    if accelerator.is_main_process:
        accelerator.print(
            f"WandB initialized in {time.time() - logger_init_start:.2f}s"
        )
        run_info[0] = logger.get_run_id()
        run_info[1] = (
            (wandb.run.name if getattr(wandb, "run", None) is not None else None)
            or run_info[0]
            or "train_run"
        )

    from accelerate.utils import broadcast_object_list

    broadcast_object_list(run_info)
    wandb_run_id, run_name = run_info

    cfg.output_dir = base_output_dir / str(run_name)

    if accelerator.is_main_process:
        cfg.output_dir.mkdir(parents=True, exist_ok=True)
        if getattr(wandb, "run", None) is not None:
            wandb.config.update(
                {"output_dir": str(cfg.output_dir)}, allow_val_change=True
            )

    accelerator.wait_for_everyone()

    model_init_start = time.time()
    model = HASST(
        in_channels=cfg.in_channels,
        out_channels=cfg.out_channels,
        embed_dim=cfg.embed_dim,
        num_blocks=cfg.num_blocks,
        lonpe_scale_physical=cfg.lonpe_scale_physical,
        lonpe_shot_range=tuple(cfg.lonpe_shot_range),
        lonpe_read_range=tuple(cfg.lonpe_read_range),
    )

    if torch.cuda.is_available() and cfg.channels_last:
        model = model.to(memory_format=torch.channels_last)
    if accelerator.is_main_process:
        accelerator.print(f"Model initialized in {time.time() - model_init_start:.2f}s")

    try:
        import importlib.util

        mamba_available = importlib.util.find_spec("mamba_ssm") is not None
    except Exception:
        mamba_available = False

    if not mamba_available:
        accelerator.print("Warning: 'mamba_ssm' package not found.")
        if cfg.require_mamba:
            raise RuntimeError(
                "Configuration requires 'mamba-ssm' but it is not installed."
            )

    criterion = CompositeLoss(cfg).to(accelerator.device)

    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        is_bias = name.endswith(".bias")
        is_low_dim = param.ndim <= 1
        is_norm = "norm" in name.lower()
        is_scale = any(x in name.lower() for x in ["beta", "gamma", "alpha"])
        is_prompt = "prompt" in name.lower()

        if is_bias or is_low_dim or is_norm or is_scale or is_prompt:
            no_decay.append(param)
        else:
            decay.append(param)

    optim_groups = [
        {"params": decay, "weight_decay": cfg.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]

    optimizer = optim.AdamW(
        optim_groups,
        lr=cfg.lr_initial,
        betas=(cfg.beta1, cfg.beta2),
        fused=True if torch.cuda.is_available() else False,
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.total_iters, eta_min=cfg.lr_min
    )

    patch_size = cfg.patch_sizes[current_phase]
    batch_size = cfg.batch_sizes[current_phase]

    train_loader, val_loader, ds_time = create_dataloaders(cfg, patch_size, batch_size)
    if accelerator.is_main_process:
        accelerator.print(f"Datasets initialized in {ds_time:.2f}s")

    model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, scheduler
    )

    try:
        unwrapped_model = get_raw_model(model, accelerator)
        if (
            accelerator.is_main_process
            and cfg.wandb_watch
            and getattr(wandb, "run", None) is not None
        ):
            wandb.watch(unwrapped_model, log="all", log_freq=cfg.wandb_watch_log_freq)
    except Exception:
        if accelerator.is_main_process:
            accelerator.print("Warning: Could not enable wandb.watch.")

    if cfg.resume and checkpoint_data is not None:
        restore_start = time.time()
        accelerator.print(
            f"Restoring weights and states from checkpoint (Step: {global_step})..."
        )

        unwrapped_model = get_raw_model(model, accelerator)
        unwrapped_model.load_state_dict(checkpoint_data["model_state_dict"])

        optimizer.load_state_dict(checkpoint_data["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint_data["scheduler_state_dict"])

        if rng_states:
            random.setstate(rng_states["python"])
            np.random.set_state(rng_states["numpy"])
            torch.set_rng_state(rng_states["torch"])
            if torch.cuda.is_available() and rng_states.get("torch_cuda") is not None:
                torch.cuda.set_rng_state_all(rng_states["torch_cuda"])

        try:
            current_lr = scheduler.get_last_lr()[0]
        except (AttributeError, TypeError, IndexError):
            current_lr = cfg.lr_initial

        if abs(current_lr - cfg.lr_initial) > 1e-9:
            accelerator.print(
                f"Re-igniting training: Updating Learning Rate to {cfg.lr_initial:.2e}"
            )
            for param_group in optimizer.param_groups:
                param_group["lr"] = cfg.lr_initial

            new_scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=cfg.total_iters - global_step, eta_min=cfg.lr_min
            )
            scheduler = accelerator.prepare(new_scheduler)

        if accelerator.is_main_process:
            accelerator.print(
                f"Weights/states restored in {time.time() - restore_start:.2f}s"
            )

    if cfg.use_compile and hasattr(torch, "compile"):
        try:
            model = torch.compile(model)
        except Exception as e:
            accelerator.print(f"torch.compile failed: {e}.")

    model.train()

    if accelerator.is_main_process:
        num_params = sum(p.numel() for p in model.parameters())
        accelerator.print(f"Total model parameters: {num_params:,}")
        if getattr(wandb, "run", None) is not None:
            wandb.config.update({"model/parameters": num_params})

    progress_bar = tqdm(
        total=cfg.total_iters,
        initial=global_step,
        disable=not accelerator.is_main_process,
        desc=f"Phase {current_phase} (Patch {patch_size})",
    )

    def _to_host_scalars(
        loss_tensor: torch.Tensor, loss_components: Dict[str, Any]
    ) -> Dict[str, float]:
        """Batch scalar extraction to reduce GPU->CPU sync points."""
        tensor_keys = []
        tensor_vals = [loss_tensor.detach()]

        for k, v in loss_components.items():
            if isinstance(v, torch.Tensor):
                tensor_keys.append(k)
                tensor_vals.append(v.detach())

        host_vals = torch.stack([t.float() for t in tensor_vals]).cpu().tolist()
        metrics = {"train/loss": host_vals[0]}

        for i, key in enumerate(tensor_keys, start=1):
            metrics[f"train/{key}"] = host_vals[i]

        for k, v in loss_components.items():
            if not isinstance(v, torch.Tensor):
                metrics[f"train/{k}"] = float(v)

        return metrics

    def get_checkpoint_state() -> Dict[str, Any]:
        """Captures the current state of training for checkpointing."""
        unwrapped_model = get_raw_model(model, accelerator)
        state = {
            "global_step": global_step,
            "current_phase": current_phase,
            "model_config": {
                "embed_dim": cfg.embed_dim,
                "num_blocks": cfg.num_blocks,
                "in_channels": cfg.in_channels,
                "out_channels": cfg.out_channels,
            },
            "model_state_dict": unwrapped_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_psnr": best_psnr,
            "wandb_run_id": wandb_run_id,
            "rng_states": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
                "torch_cuda": torch.cuda.get_rng_state_all()
                if torch.cuda.is_available()
                else None,
            },
        }
        return state

    def save_checkpoint(path: Union[str, Path], is_best: bool = False) -> None:
        """Saves a training checkpoint.

        Args:
            path: Path to save the checkpoint file.
            is_best: Whether this is the best model so far.
        """
        if not accelerator.is_main_process:
            return
        checkpoint_state = get_checkpoint_state()
        torch.save(checkpoint_state, path)

        last_path = cfg.output_dir / "last_model.pth"
        if path != last_path:
            torch.save(checkpoint_state, last_path)

    batch_start_time = time.time()
    optimizer.zero_grad(set_to_none=True)
    try:
        while global_step < cfg.total_iters:
            should_stop = torch.tensor(0, device=accelerator.device)
            if accelerator.is_main_process:
                if max_seconds and (time.time() - start_time > max_seconds):
                    should_stop += 1

            if accelerator.num_processes > 1:
                should_stop = accelerator.reduce(should_stop, reduction="sum")

            if should_stop > 0:
                accelerator.print("\nTime limit reached. Synchronized shutdown...")
                break

            for noisy, gt in train_loader:
                if global_step >= cfg.total_iters:
                    break

                with accelerator.accumulate(model):
                    if random.random() < 0.5:
                        noisy, gt = adversarial_frequency_mixup(
                            noisy, gt, alpha=random.uniform(0.1, 0.4)
                        )
                    if random.random() < 0.5:
                        noisy, gt = apply_noise_cutmix(noisy, gt)

                    noisy = torch.clamp(noisy, 0.0, 1.0)

                    if torch.cuda.is_available() and cfg.channels_last:
                        noisy = noisy.contiguous(memory_format=torch.channels_last)
                        gt = gt.contiguous(memory_format=torch.channels_last)

                    with accelerator.autocast():
                        pred = model(noisy)
                        loss, loss_dict = criterion(pred, gt)

                    accelerator.backward(loss)

                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)

                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)

                logged_this_step = False

                if (
                    global_step % cfg.log_freq == 0
                    and "loss" in locals()
                    and "loss_dict" in locals()
                ):
                    elapsed = time.time() - batch_start_time
                    img_per_sec = (
                        (
                            cfg.batch_sizes[current_phase]
                            * accelerator.num_processes
                            * cfg.log_freq
                        )
                        / elapsed
                        if elapsed > 0
                        else 0
                    )
                    gpu_mem_gb = (
                        torch.cuda.max_memory_reserved() / (1024**3)
                        if torch.cuda.is_available()
                        else 0
                    )

                    log_data = _to_host_scalars(loss, loss_dict)
                    log_data["train/learning_rate"] = scheduler.get_last_lr()[0]
                    log_data["train/patch_size"] = patch_size

                    logger.log_metrics(global_step, log_data, commit=False)
                    logger.log_system_metrics(
                        global_step, img_per_sec, gpu_mem_gb, commit=False
                    )

                    batch_start_time = time.time()
                    logged_this_step = True

                if global_step > 0 and global_step % cfg.val_freq == 0:
                    val_psnr, val_ssim, val_sample = evaluate_pipeline(
                        model, val_loader, accelerator
                    )
                    logger.log_metrics(
                        global_step,
                        {"val/psnr": val_psnr, "val/ssim": val_ssim},
                        commit=False,
                    )
                    logger.log_gradients(global_step, model, commit=False)

                    if val_sample is not None:
                        logger.log_visual_artifacts(
                            global_step, *val_sample, prefix="visuals_val", commit=False
                        )

                    if val_psnr > best_psnr:
                        best_psnr = val_psnr
                        if accelerator.is_main_process:
                            best_path = cfg.output_dir / "best_model.pth"
                            save_checkpoint(best_path, is_best=True)
                            progress_bar.write(
                                f"Step {global_step}: New Best Model! PSNR: {best_psnr:.2f}"
                            )

                    logged_this_step = True

                if logged_this_step:
                    logger.log_metrics(global_step, {}, commit=True)

                if global_step > 0 and global_step % cfg.checkpoint_freq == 0:
                    save_checkpoint(cfg.output_dir / "last_model.pth")

                global_step += 1
                progress_bar.update(1)

                if (
                    current_phase < len(cfg.phase_milestones)
                    and global_step == cfg.phase_milestones[current_phase]
                ):
                    current_phase += 1

                    if current_phase >= len(cfg.patch_sizes) or current_phase >= len(
                        cfg.batch_sizes
                    ):
                        current_phase -= 1
                        continue

                    patch_size = cfg.patch_sizes[current_phase]
                    batch_size = cfg.batch_sizes[current_phase]

                    accelerator.print(
                        f"\nScaling up! Phase {current_phase}: Patch {patch_size}x{patch_size}, Batch {batch_size}"
                    )
                    progress_bar.set_description(
                        f"Phase {current_phase} (Patch {patch_size})"
                    )

                    accelerator.wait_for_everyone()
                    accelerator.free_memory()
                    train_loader, val_loader, ds_time = create_dataloaders(
                        cfg, patch_size, batch_size
                    )
                    train_loader, val_loader = accelerator.prepare(
                        train_loader, val_loader
                    )
                    break

                if global_step % cfg.log_freq == 0:
                    should_stop = torch.tensor(0, device=accelerator.device)
                    if accelerator.is_main_process:
                        if max_seconds and (time.time() - start_time > max_seconds):
                            should_stop += 1

                    if accelerator.num_processes > 1:
                        should_stop = accelerator.reduce(should_stop, reduction="sum")

                    if should_stop > 0:
                        break
    finally:
        progress_bar.close()
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            accelerator.print("\nFinalizing training...")

            last_path = cfg.output_dir / "last_model.pth"
            save_checkpoint(last_path)

            logger.log_model_artifact(
                last_path, "model-checkpoints", metadata={"step": global_step}
            )

            best_path = cfg.output_dir / "best_model.pth"
            if best_path.exists():
                logger.log_model_artifact(
                    best_path,
                    "best-model",
                    metadata={"psnr": best_psnr, "step": global_step},
                )

            logger.finish()


def main() -> None:
    """Main entry point for the training script."""
    parser = argparse.ArgumentParser(description="HASST Multi-Config Training Script")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to a single YAML config, or a directory containing multiple YAML configs.",
    )
    args = parser.parse_args()

    config_paths = []
    if Path(args.config).is_dir():
        config_paths = sorted(list(Path(args.config).glob("*.yaml")))
    else:
        config_paths = [Path(args.config)]

    if not config_paths:
        logger_cli.error(f"No configuration files found at {args.config}")
        return

    logger_cli.info(f"Starting multi-config run: {len(config_paths)} configs found.")

    start_time = time.time()
    for path in config_paths:
        logger_cli.info(f"Processing config: {path}")
        try:
            cfg = Config.load_from_yaml(str(path))
            run_training(cfg, start_time=start_time)
        except Exception as e:
            import traceback

            traceback.print_exc()
            logger_cli.error(f"Failed to process config {path}: {e}")
            continue

    logger_cli.info("All configurations processed.")


if __name__ == "__main__":
    main()
