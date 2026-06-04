import argparse
import base64
import csv
import io
import zipfile
from pathlib import Path

import numpy as np
import scipy.io
import torch
from tqdm import tqdm

from models.hasst import HASST
from models.inference import HASSTInferenceEngine


def load_checkpoint_file(model_path, allow_unsafe_pickle=True):
    """Loads checkpoints across PyTorch versions, including 2.6+ weights_only default changes."""
    try:
        return torch.load(model_path, map_location="cpu")
    except TypeError:
        # Older torch versions may not support modern kwargs behavior.
        return torch.load(model_path, map_location="cpu")
    except Exception as exc:
        msg = str(exc)
        if allow_unsafe_pickle and "Weights only load failed" in msg:
            print(
                "Checkpoint uses pickled non-tensor objects. "
                "Retrying with weights_only=False (trusted source required)."
            )
            return torch.load(model_path, map_location="cpu", weights_only=False)
        raise


def load_benchmark_blocks(benchmark_path):
    """Loads BenchmarkNoisyBlocksSrgb from .mat or .zip containing a .mat file."""
    if str(benchmark_path).lower().endswith(".zip"):
        with zipfile.ZipFile(benchmark_path, "r") as zf:
            mat_files = [name for name in zf.namelist() if name.lower().endswith(".mat")]
            if not mat_files:
                raise ValueError(f"No .mat file found inside zip: {benchmark_path}")
            if len(mat_files) > 1:
                print(f"Multiple .mat files found in zip, using first: {mat_files[0]}")
            with zf.open(mat_files[0], "r") as f:
                mat_data = scipy.io.loadmat(io.BytesIO(f.read()))
    else:
        mat_data = scipy.io.loadmat(benchmark_path)

    if "BenchmarkNoisyBlocksSrgb" not in mat_data:
        raise KeyError("Variable 'BenchmarkNoisyBlocksSrgb' not found in benchmark file.")

    return mat_data["BenchmarkNoisyBlocksSrgb"]


def _has_mamba_weights(state_dict):
    return any(".mamba." in k for k in state_dict.keys())


def array_to_base64string(x):
    """Encodes a uint8 image block into a UTF-8 base64 string."""
    return base64.b64encode(x.tobytes()).decode("utf-8")


def _iter_submission_blocks(denoised_blocks):
    block_id = 0
    for i in range(denoised_blocks.shape[0]):
        for j in range(denoised_blocks.shape[1]):
            yield block_id, denoised_blocks[i, j]
            block_id += 1


def save_submission_csv(denoised_blocks, output_path):
    """Saves SIDD submission in Kaggle CSV format: ID,BLOCK(base64)."""
    output_path = Path(output_path)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["ID", "BLOCK"])
        for block_id, block in _iter_submission_blocks(denoised_blocks):
            writer.writerow([block_id, array_to_base64string(block)])


def save_submission_zip_csv(denoised_blocks, output_zip_path):
    """Saves a zip archive that contains exactly one CSV submission file."""
    output_zip_path = Path(output_zip_path)
    csv_name = f"{output_zip_path.stem}.csv"
    with zipfile.ZipFile(output_zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        with zf.open(csv_name, "w") as f_bin:
            with io.TextIOWrapper(f_bin, encoding="utf-8", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["ID", "BLOCK"])
                for block_id, block in _iter_submission_blocks(denoised_blocks):
                    writer.writerow([block_id, array_to_base64string(block)])


def _load_model_weights(model, clean_state_dict, allow_missing_mamba=False):
    try:
        model.load_state_dict(clean_state_dict, strict=True)
        return
    except RuntimeError as exc:
        msg = str(exc)
        has_mamba = _has_mamba_weights(clean_state_dict)
        looks_like_missing_mamba = "Unexpected key(s) in state_dict" in msg and has_mamba

        if looks_like_missing_mamba and not allow_missing_mamba:
            raise RuntimeError(
                "Checkpoint contains Mamba weights, but current runtime does not expose matching Mamba modules. "
                "This usually means 'mamba-ssm' / 'causal-conv1d' are not installed in the inference environment.\n"
                "Install them and rerun for faithful PSNR:\n"
                "  uv sync --extra mamba\n"
                "or:\n"
                "  pip install mamba-ssm causal-conv1d\n"
                "If you intentionally want degraded fallback inference, rerun with --allow_missing_mamba."
            ) from exc

        if allow_missing_mamba:
            missing, unexpected = model.load_state_dict(clean_state_dict, strict=False)
            print(
                "Warning: Loaded checkpoint with strict=False. "
                f"Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}"
            )
            if has_mamba:
                print(
                    "Warning: Mamba weights were skipped. Global sequence branch is disabled, "
                    "so PSNR will be lower than a fully matched environment."
                )
            return

        raise


def predict_benchmark(model_path, benchmark_path, output_path, use_tta=True, allow_missing_mamba=False):
    """
    Inference script for the official SIDD Benchmark (sRGB).
    Loads BenchmarkNoisyBlocksSrgb and saves either:
    - .csv (Kaggle submission format: ID,BLOCK base64)
    - .zip (containing a .csv submission file)
    - .mat (legacy format, not for Kaggle submission page)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load Model
    print(f"Loading checkpoint from {model_path}...")
    checkpoint = load_checkpoint_file(model_path, allow_unsafe_pickle=True)

    # Extract config from checkpoint if available to avoid hardcoding
    if "model_config" in checkpoint:
        m_cfg = checkpoint["model_config"]
        print(f"Found model config in checkpoint: {m_cfg}")
        model = HASST(
            embed_dim=m_cfg.get("embed_dim", 64),
            num_blocks=m_cfg.get("num_blocks", 4),
            in_channels=m_cfg.get("in_channels", 3),
            out_channels=m_cfg.get("out_channels", 3)
        )
    else:
        print("Warning: No model config found in checkpoint. Using defaults (embed_dim=64, num_blocks=4).")
        model = HASST(embed_dim=64, num_blocks=4)
    
    # Handle state_dict key variations (e.g. from Accelerator or torch.compile)
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    
    # Remove 'module.' or '_orig_mod.' prefixes if present
    clean_state_dict = {}
    for k, v in state_dict.items():
        name = k.replace("module.", "").replace("_orig_mod.", "")
        clean_state_dict[name] = v
        
    _load_model_weights(model, clean_state_dict, allow_missing_mamba=allow_missing_mamba)
    model = model.to(device)
    model.eval()

    # 2. Load Benchmark Data
    print(f"Loading benchmark file: {benchmark_path}")
    noisy_blocks = load_benchmark_blocks(benchmark_path)  # Shape: (40, 32, 256, 256, 3)

    num_scenes, num_blocks, H, W, C = noisy_blocks.shape
    print(f"Found {num_scenes} scenes with {num_blocks} blocks each ({H}x{W}px)")

    # 3. Initialize Inference Engine
    engine = HASSTInferenceEngine(model, device=device)

    # 4. Process Blocks
    denoised_blocks = np.zeros_like(noisy_blocks, dtype=np.uint8)

    for s in range(num_scenes):
        print(f"Processing Scene {s+1}/{num_scenes}...")
        for b in tqdm(range(num_blocks), desc=f"Scene {s+1}", leave=False):
            # Preprocess: (H, W, C) [0, 255] -> (1, C, H, W) [0, 1]
            block = noisy_blocks[s, b].astype(np.float32) / 255.0
            block_tensor = torch.from_numpy(block).permute(2, 0, 1).unsqueeze(0).to(device)

            # Inference
            with torch.no_grad():
                if use_tta:
                    # 8x Geometric Self-Ensemble
                    pred_tensor = engine.forward_tta(block_tensor)
                else:
                    # Single Forward Pass
                    pred_tensor = model(block_tensor)

            # Postprocess: (1, C, H, W) [0, 1] -> (H, W, C) [0, 255]
            pred_np = pred_tensor.squeeze(0).permute(1, 2, 0).cpu().clamp(0.0, 1.0).numpy()
            denoised_blocks[s, b] = (pred_np * 255.0).round().astype(np.uint8)

    # 5. Save Results
    print(f"Saving results to {output_path}...")
    output_suffix = Path(output_path).suffix.lower()

    if output_suffix == ".csv":
        save_submission_csv(denoised_blocks, output_path)
    elif output_suffix == ".zip":
        save_submission_zip_csv(denoised_blocks, output_path)
    elif output_suffix == ".mat":
        # Legacy format used by some scripts. Kaggle submission page expects CSV/Parquet.
        scipy.io.savemat(output_path, {"DenoisedBlocksSrgb": denoised_blocks})
    else:
        raise ValueError(
            f"Unsupported output extension '{output_suffix}'. "
            "Use .csv (recommended), .zip (with csv inside), or .mat (legacy)."
        )

    print("Prediction complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict SIDD Benchmark Blocks")
    parser.add_argument("--model", type=str, required=True, help="Path to best_model.pth")
    parser.add_argument("--benchmark", type=str, required=True, help="Path to BenchmarkNoisyBlocksSrgb.mat")
    parser.add_argument(
        "--output",
        type=str,
        default="SubmitSrgb.csv",
        help="Output filename (.csv recommended for Kaggle submission)",
    )
    parser.add_argument("--no_tta", action="store_true", help="Disable Test-Time Augmentation (faster but lower PSNR)")
    parser.add_argument(
        "--allow_missing_mamba",
        action="store_true",
        help="Allow loading checkpoints even if Mamba weights cannot be mapped (degraded fallback).",
    )

    args = parser.parse_args()
    
    # Ensure src is in PYTHONPATH if running from root
    predict_benchmark(
        args.model,
        args.benchmark,
        args.output,
        use_tta=not args.no_tta,
        allow_missing_mamba=args.allow_missing_mamba,
    )
