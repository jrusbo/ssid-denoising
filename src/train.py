import argparse
import inspect
import json
import logging
import random
import time
from dataclasses import asdict
from pathlib import Path

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
from data.augmentations import adversarial_frequency_mixup, apply_noise_cutmix, reset_augmentation_buffers
from losses.loss import CompositeLoss
from models.hasst import HASST
from utils.logger import WandBValidationLogger
from utils.metrics import compute_psnr, compute_ssim


# Setup logging for multi-config runs
logging.basicConfig(level=logging.INFO)
logger_cli = logging.getLogger(__name__)


def make_worker_init_fn(base_seed: int, worker_cpu_threads: int):
    """Create deterministic per-worker setup without CPU thread oversubscription."""

    def _worker_init_fn(worker_id: int):
        worker_seed = base_seed + worker_id
        random.seed(worker_seed)
        np.random.seed(worker_seed % (2**32 - 1))
        torch.manual_seed(worker_seed)

        if worker_cpu_threads and worker_cpu_threads > 0:
            torch.set_num_threads(worker_cpu_threads)
            try:
                torch.set_num_interop_threads(1)
            except RuntimeError:
                # set_num_interop_threads may be immutable depending on worker startup path.
                pass

    return _worker_init_fn


def get_raw_model(model, accelerator):
    """
    Unwraps model from Accelerator and torch.compile for saving.
    Handles nested wrapping recursively to ensure we get the base nn.Module.
    """
    while hasattr(model, "module") or hasattr(model, "_orig_mod"):
        if hasattr(model, "module"):
            model = model.module
        if hasattr(model, "_orig_mod"):
            model = model._orig_mod
    
    # Final safety check with accelerator.unwrap_model (handles more cases)
    return accelerator.unwrap_model(model)


@torch.no_grad()
def evaluate_pipeline(model, dataloader, accelerator):
    """
    Evaluates the model on validation sample sets.
    Computes metrics locally and averages them across processes.
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
        mininterval=2.0, # Reduce tqdm update frequency for better Colab performance
    )
    
    # Use autocast to match training speed and precision
    with accelerator.autocast():
        for i, (noisy, gt) in enumerate(pbar):
            pred = model(noisy)
            
            # Save one sample for visualization (from the main process)
            if i == 0 and accelerator.is_main_process:
                noise_prior = None
                if hasattr(raw_model, "estimate_noise_prior"):
                    noise_prior = raw_model.estimate_noise_prior(noisy).detach()
                val_sample = (noisy.detach(), pred.detach(), gt.detach(), noise_prior)

            # Batch parsing metric accumulations (Calculated locally per GPU)
            # compute_psnr and compute_ssim now return tensors to avoid GPU-CPU syncs
            local_psnr += compute_psnr(pred, gt)
            local_ssim += compute_ssim(pred, gt, size_average=False)
            local_samples += pred.shape[0]

    # Sync and reduce across all processes (No-op on single GPU)
    metrics = torch.stack([local_psnr, local_ssim, local_samples])

    if accelerator.num_processes > 1:
        metrics = accelerator.reduce(metrics, reduction="sum")
    
    global_psnr, global_ssim, total_samples = metrics.tolist()

    model.train()
    if total_samples == 0:
        return 0.0, 0.0, None
    return global_psnr / total_samples, global_ssim / total_samples, val_sample


def create_dataloaders(cfg, patch_size, batch_size):
    """Dynamically creates dataloaders for the progressive learning schedule."""
    start_time = time.time()
    train_dataset = SIDDDatasetLMDB(
        lmdb_dir=cfg.lmdb_dir, patch_size=patch_size, split="train", seed=cfg.seed
    )
    val_dataset = SIDDDatasetLMDB(
        lmdb_dir=cfg.lmdb_dir, patch_size=patch_size, split="val", seed=cfg.seed
    )
    dataset_time = time.time() - start_time
    dataloader_signature = inspect.signature(DataLoader).parameters

    train_loader_kwargs = dict(
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
        # Disable strict batch ordering to avoid head-of-line blocking from a slow worker.
        train_loader_kwargs["in_order"] = cfg.train_loader_in_order

    if (
        "pin_memory_device" in dataloader_signature
        and cfg.pin_memory
        and torch.cuda.is_available()
    ):
        train_loader_kwargs["pin_memory_device"] = f"cuda:{torch.cuda.current_device()}"

    train_loader = DataLoader(**train_loader_kwargs)

    # Validation loader: We use a small number of workers (1 or 2) but disable persistent_workers.
    # This prevents worker-shutdown deadlocks on Colab while keeping data loading fast.
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


def run_training(cfg: Config, start_time: float = None):
    # 0. Set seed for initial reproducibility
    init_start = time.time()
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)
    
    # Reset global augmentation buffers to prevent cross-run contamination
    reset_augmentation_buffers()

    # 1. Initialize HuggingFace Accelerator
    accelerator = Accelerator(
        mixed_precision=cfg.mixed_precision,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
    )

    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = cfg.cudnn_benchmark

    # 2. Print Configuration for Verification
    if accelerator.is_main_process:
        accelerator.print(f"Accelerator initialized in {time.time() - init_start:.2f}s")
        accelerator.print("\n" + "="*50)
        accelerator.print("RECEIVED CONFIGURATION:")
        # ...
        cfg_dict = asdict(cfg)
        for k, v in cfg_dict.items():
            if isinstance(v, (str, Path)) and any(x in k for x in ["dir", "path"]):
                try:
                    cfg_dict[k] = str(Path(v).resolve())
                except Exception:
                    pass
        accelerator.print(json.dumps(cfg_dict, indent=4))
        accelerator.print("="*50 + "\n")

    # Start timer for Kaggle limit
    if start_time is None:
        start_time = time.time()
    max_seconds = cfg.max_hours * 3600 if cfg.max_hours and cfg.max_hours > 0 else None

    # --- Resuming Mechanism (Pre-Logger) ---
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
            # Search for the newest last_model.pth in any subdirectory
            last_models = list(base_output_dir.glob("*/last_model.pth"))
            if last_models:
                ckpt_path = sorted(last_models, key=lambda x: x.stat().st_mtime)[-1]

        if ckpt_path:
            # If a directory is provided, look for last_model.pth inside it
            if ckpt_path.is_dir():
                ckpt_path = ckpt_path / "last_model.pth"
            
            if not ckpt_path.exists():
                accelerator.print(f"Warning: Checkpoint file {ckpt_path} not found. Starting from scratch.")
                ckpt_path = None
            else:
                accelerator.print(f"Resuming from {ckpt_path}...")
                try:
                    # We use weights_only=False because we need to load RNG states, 
                    # optimizer states, and scheduler states which are not simple weights.
                    checkpoint_data = torch.load(ckpt_path, map_location="cpu", weights_only=False)
                except TypeError:
                    # Fallback for older PyTorch versions that don't have weights_only
                    checkpoint_data = torch.load(ckpt_path, map_location="cpu")
            
            if ckpt_path:
                accelerator.print(f"Checkpoint loaded in {time.time() - ckpt_load_start:.2f}s")
                # Load basic states
                global_step = checkpoint_data["global_step"]
                current_phase = checkpoint_data.get("current_phase", 0)
                best_psnr = checkpoint_data.get("best_psnr", 0.0)
                
                # Prioritize wandb_run_id from checkpoint, then config
                wandb_run_id = checkpoint_data.get("wandb_run_id") or cfg.wandb_run_id
                rng_states = checkpoint_data.get("rng_states")

                if wandb_run_id:
                    accelerator.print(f"Found WandB Run ID to resume: {wandb_run_id}")

                # Update current phase based on global_step
                while (
                    current_phase < len(cfg.phase_milestones)
                    and global_step >= cfg.phase_milestones[current_phase]
                ):
                    current_phase += 1

                # Safety: Clamp current_phase to the number of available configurations
                current_phase = min(current_phase, len(cfg.patch_sizes) - 1, len(cfg.batch_sizes) - 1)
        else:
            accelerator.print("Warning: 'resume' is True but no valid checkpoint was found. Starting from scratch.")
            wandb_run_id = cfg.wandb_run_id

    # 2. Initialize WandB Logger (with potential run_id)
    if wandb_run_id and accelerator.is_main_process:
        accelerator.print(f"Attempting to resume WandB run: {wandb_run_id}")
    
    logger_init_start = time.time()
    logger = WandBValidationLogger(cfg, is_main_process=accelerator.is_main_process, run_id=wandb_run_id)
    
    # Update wandb_run_id and ensure it's synced across all processes
    # We use broadcast_object_list to ensure all workers have the same Run ID and Name
    run_info = [None, None]
    if accelerator.is_main_process:
        accelerator.print(f"WandB initialized in {time.time() - logger_init_start:.2f}s")
        run_info[0] = logger.get_run_id()
        run_info[1] = (wandb.run.name if getattr(wandb, "run", None) is not None else None) or run_info[0] or "train_run"

    from accelerate.utils import broadcast_object_list
    broadcast_object_list(run_info)
    wandb_run_id, run_name = run_info
    
    # Update cfg.output_dir to be run-specific (Using human-readable run name)
    cfg.output_dir = base_output_dir / str(run_name)
    
    if accelerator.is_main_process:
        cfg.output_dir.mkdir(parents=True, exist_ok=True)
        # Update wandb config with the final run-specific output directory
        if getattr(wandb, "run", None) is not None:
            wandb.config.update({"output_dir": str(cfg.output_dir)}, allow_val_change=True)

    accelerator.wait_for_everyone()

    # 3. Model, Loss, and Optimizer Setup
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

    # -- Dependency sanity check for the Mamba attentive state-space implementation
    try:
        import importlib.util
        _mamba_spec = importlib.util.find_spec("mamba_ssm")
        mamba_available = _mamba_spec is not None
    except Exception:
        mamba_available = False

    if not mamba_available:
        accelerator.print("Warning: 'mamba_ssm' package not found. The AttentiveStateSpaceBlock will fallback to a zeroed global branch. To match the HASST report, install 'mamba-ssm' and 'causal-conv1d'.")
        if cfg.require_mamba:
            raise RuntimeError("Configuration requires 'mamba-ssm' but it is not installed. Install mamba-ssm and causal-conv1d or set Config.require_mamba=False.")

    criterion = CompositeLoss(cfg).to(accelerator.device)

    # Separate parameters so AdamW doesn't apply weight decay to biases and LayerNorms
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        
        # We exclude from weight decay:
        # 1. Biases (usually 1D)
        # 2. Norm weights/biases (can be 1D or 4D like in LayerNorm2d)
        # 3. Scale parameters like beta, gamma (usually 4D (1, C, 1, 1))
        # 4. Positional embeddings or prompts
        
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

    # 4. Prepare Phase Dataloaders
    patch_size = cfg.patch_sizes[current_phase]
    batch_size = cfg.batch_sizes[current_phase]

    train_loader, val_loader, ds_time = create_dataloaders(cfg, patch_size, batch_size)
    if accelerator.is_main_process:
        accelerator.print(f"Datasets initialized in {ds_time:.2f}s")
    
    # 5. Prepare everything with Accelerate
    # NOTE: We prepare BEFORE loading state dicts to ensure the wrapped objects 
    # receive the states correctly.
    prep_start = time.time()
    model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, scheduler
    )
    if accelerator.is_main_process:
        accelerator.print(f"Accelerator.prepare completed in {time.time() - prep_start:.2f}s")

    # If requested, enable wandb.watch on the unwrapped model to collect histograms.
    try:
        unwrapped_model = get_raw_model(model, accelerator)
        if accelerator.is_main_process and cfg.wandb_watch and getattr(wandb, "run", None) is not None:
            try:
                wandb.watch(unwrapped_model, log="all", log_freq=cfg.wandb_watch_log_freq)
                accelerator.print(f"wandb.watch enabled (log_freq={cfg.wandb_watch_log_freq}).")
            except Exception as e:
                accelerator.print(f"wandb.watch failed: {e}. Continuing without detailed watch.")
    except Exception:
        # If anything goes wrong with unwrapping or wandb, continue but log a message.
        if accelerator.is_main_process:
            accelerator.print("Warning: Could not enable wandb.watch for the model (unwrap/watch failure).")

    # Now load weights if resuming (Into PREPARED objects)
    if cfg.resume and checkpoint_data is not None:
        restore_start = time.time()
        accelerator.print(f"Restoring weights and states from checkpoint (Step: {global_step})...")
        
        # Load model weights into the unwrapped model
        unwrapped_model = get_raw_model(model, accelerator)
        unwrapped_model.load_state_dict(checkpoint_data["model_state_dict"])
        
        # Load optimizer and scheduler states
        optimizer.load_state_dict(checkpoint_data["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint_data["scheduler_state_dict"])

        # Restore RNG states for full reproducibility
        if rng_states:
            random.setstate(rng_states["python"])
            np.random.set_state(rng_states["numpy"])
            torch.set_rng_state(rng_states["torch"])
            if torch.cuda.is_available() and rng_states.get("torch_cuda") is not None:
                torch.cuda.set_rng_state_all(rng_states["torch_cuda"])
            accelerator.print("Exact RNG states restored.")
        
        # Verify LR restoration
        current_lr = scheduler.get_last_lr()[0]
        accelerator.print(f"Restored Learning Rate: {current_lr:.2e}")
        if accelerator.is_main_process:
            accelerator.print(f"Weights/states restored in {time.time() - restore_start:.2f}s")

    # 6. Apply torch.compile
    if cfg.use_compile and hasattr(torch, "compile"):
        compile_start = time.time()
        accelerator.print("Compiling model with torch.compile...")
        try:
            model = torch.compile(model)
            if accelerator.is_main_process:
                accelerator.print(f"torch.compile call completed in {time.time() - compile_start:.2f}s (Note: Real compilation happens at first forward pass)")
        except Exception as e:
            accelerator.print(f"torch.compile failed: {e}. Falling back to standard execution.")

    # 7. Training Loop
    model.train()
    
    if accelerator.is_main_process:
        num_params = sum(p.numel() for p in model.parameters())
        accelerator.print(f"Total model parameters: {num_params:,}")
        if getattr(wandb, "run", None) is not None:
            wandb.config.update({"model/parameters": num_params})
        accelerator.print(f"Full initialization took {time.time() - init_start:.2f}s")

    progress_bar = tqdm(
        total=cfg.total_iters,
        initial=global_step,
        disable=not accelerator.is_main_process,
        desc=f"Phase {current_phase} (Patch {patch_size})"
    )

    def _to_host_scalars(loss_tensor, loss_components):
        """Batch scalar extraction to reduce GPU->CPU sync points in logging."""
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

    def get_checkpoint_state():
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
                "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            }
        }
        return state

    def save_checkpoint(path, is_best=False):
        if not accelerator.is_main_process:
            return
        checkpoint_state = get_checkpoint_state()
        torch.save(checkpoint_state, path)
        
        # Always overwrite a 'last_model.pth' for easy resuming
        last_path = cfg.output_dir / "last_model.pth"
        if path != last_path:
            torch.save(checkpoint_state, last_path)

    batch_start_time = time.time()
    optimizer.zero_grad(set_to_none=True)
    try:
        while global_step < cfg.total_iters:
            # Check for time limit reached (Synchronized across all processes)
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
                    # Apply advanced augmentations on GPU
                    if random.random() < 0.5:
                        noisy, gt = adversarial_frequency_mixup(noisy, gt, alpha=random.uniform(0.1, 0.4))
                    if random.random() < 0.5:
                        noisy, gt = apply_noise_cutmix(noisy, gt)

                    # Ensure augmented inputs are clamped to [0, 1] range
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

                # Logging Logic
                logged_this_step = False

                if global_step % cfg.log_freq == 0:
                    elapsed = time.time() - batch_start_time
                    img_per_sec = (cfg.batch_sizes[current_phase] * accelerator.num_processes * cfg.log_freq) / elapsed if elapsed > 0 else 0
                    gpu_mem_gb = torch.cuda.max_memory_reserved() / (1024**3) if torch.cuda.is_available() else 0

                    log_data = _to_host_scalars(loss, loss_dict)
                    log_data["train/learning_rate"] = scheduler.get_last_lr()[0]
                    log_data["train/patch_size"] = patch_size
                    
                    logger.log_metrics(global_step, log_data, commit=False)
                    # Removed log_gradients from here to avoid pipeline stall
                    logger.log_system_metrics(global_step, img_per_sec, gpu_mem_gb, commit=False)
                    
                    batch_start_time = time.time()
                    logged_this_step = True

                # Validation Logic
                if global_step > 0 and global_step % cfg.val_freq == 0:
                    val_psnr, val_ssim, val_sample = evaluate_pipeline(model, val_loader, accelerator)
                    logger.log_metrics(
                        global_step,
                        {"val/psnr": val_psnr, "val/ssim": val_ssim},
                        commit=False,
                    )
                    # Gradient logging moved here (already syncing for validation)
                    logger.log_gradients(global_step, model, commit=False)

                    if val_sample is not None:
                        logger.log_visual_artifacts(global_step, *val_sample, prefix="visuals_val", commit=False)

                    if val_psnr > best_psnr:
                        best_psnr = val_psnr
                        if accelerator.is_main_process:
                            best_path = cfg.output_dir / "best_model.pth"
                            save_checkpoint(best_path, is_best=True)
                            progress_bar.write(f"Step {global_step}: New Best Model! PSNR: {best_psnr:.2f}")
                    
                    logged_this_step = True

                if logged_this_step:
                    logger.log_metrics(global_step, {}, commit=True)

                # Periodic Checkpoint (Updates last_model.pth for resuming)
                if global_step > 0 and global_step % cfg.checkpoint_freq == 0:
                    save_checkpoint(cfg.output_dir / "last_model.pth")

                global_step += 1
                progress_bar.update(1)

                # Phase Transition
                if (
                    current_phase < len(cfg.phase_milestones)
                    and global_step == cfg.phase_milestones[current_phase]
                ):
                    current_phase += 1
                    
                    if current_phase >= len(cfg.patch_sizes) or current_phase >= len(cfg.batch_sizes):
                        accelerator.print(f"Warning: Phase milestone reached but no more config for phase {current_phase}. Staying at current phase.")
                        current_phase -= 1
                        continue

                    patch_size = cfg.patch_sizes[current_phase]
                    batch_size = cfg.batch_sizes[current_phase]

                    accelerator.print(f"\nScaling up! Phase {current_phase}: Patch {patch_size}x{patch_size}, Batch {batch_size}")
                    progress_bar.set_description(f"Phase {current_phase} (Patch {patch_size})")

                    accelerator.wait_for_everyone()
                    accelerator.free_memory()
                    train_loader, val_loader, ds_time = create_dataloaders(cfg, patch_size, batch_size)
                    train_loader, val_loader = accelerator.prepare(train_loader, val_loader)
                    accelerator.print(f"Phase {current_phase} datasets initialized in {ds_time:.2f}s")
                    break

                # Check for time limit reached (Synchronized)
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
        # 8. Shutdown & Final Save
        progress_bar.close()
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            accelerator.print("\nFinalizing training and uploading artifacts...")
            
            # Save the final state locally (overwriting last_model.pth)
            last_path = cfg.output_dir / "last_model.pth"
            save_checkpoint(last_path)

            # Upload to WandB Artifacts (Only at the end to prevent v0-v100 clutter)
            # We use consistent names so WandB versions them (v1, v2, v3...) instead of creating new ones
            
            # 1. Upload 'model-checkpoints' (The most recent state, used for resuming)
            logger.log_model_artifact(last_path, "model-checkpoints", metadata={"step": global_step})
            
            # 2. Upload 'best-model' (The highest PSNR achieved)
            best_path = cfg.output_dir / "best_model.pth"
            if best_path.exists():
                logger.log_model_artifact(best_path, "best-model", metadata={"psnr": best_psnr, "step": global_step})
            
            if global_step >= cfg.total_iters:
                accelerator.print("Training Complete!")
            
            logger.finish()


def main():
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
            logger_cli.error(f"Failed to process config {path}: {e}")
            continue

    logger_cli.info("All configurations processed.")


if __name__ == "__main__":
    main()
