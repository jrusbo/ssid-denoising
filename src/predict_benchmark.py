import argparse
import base64
import csv
import os
import zipfile

import numpy as np
import scipy.io
import torch
from tqdm import tqdm

from models.hasst import HASST
from models.inference import HASSTInferenceEngine


def array_to_base64string(x):
    """Converts a numpy array to a base64 encoded string."""
    array_bytes = x.tobytes()
    base64_bytes = base64.b64encode(array_bytes)
    base64_string = base64_bytes.decode("utf-8")
    return base64_string


def _load_model_weights(model, state_dict, allow_missing_mamba=False):
    """
    Surgically loads weights, handling missing Mamba dependencies if requested.
    """
    model_keys = set(model.state_dict().keys())
    
    # Filter state_dict to only include keys that exist in the model
    # This is crucial when loading a Mamba checkpoint into a model without Mamba modules
    filtered_state_dict = {}
    missing_mamba_keys = []
    
    for k, v in state_dict.items():
        if k in model_keys:
            filtered_state_dict[k] = v
        elif ".mamba." in k:
            missing_mamba_keys.append(k)
    
    if missing_mamba_keys and allow_missing_mamba:
        print(f"Skipping {len(missing_mamba_keys)} Mamba-related weights (allow_missing_mamba=True)")
    
    msg = model.load_state_dict(filtered_state_dict, strict=not allow_missing_mamba)
    if msg.missing_keys or msg.unexpected_keys:
        print(f"Checkpoint Load Info: {msg}")


def predict_benchmark(model_path, benchmark_path, output_path, use_tta=True, patch_size=256, allow_missing_mamba=False):
    """
    Inference script for the official SIDD Benchmark (sRGB).
    Loads BenchmarkNoisyBlocksSrgb.mat and saves SubmitSrgb.csv (and .mat).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 0. Strict Dependency Check for Mamba
    try:
        import mamba_ssm
        print("Dependency 'mamba_ssm' found. Global branch active.")
    except ImportError:
        print("\n" + "!" * 80)
        print("CRITICAL WARNING: 'mamba_ssm' not found!")
        print("The AttentiveStateSpaceBlock will ZERO OUT its global branch.")
        print("This will cause a significant drop in PSNR (approx ~1.0 dB).")
        print("Please install it with: pip install mamba-ssm causal-conv1d")
        print("!" * 80 + "\n")

    # 1. Load Model
    print(f"Loading checkpoint from {model_path}...")
    try:
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    except TypeError:
        # Fallback for older torch versions that don't have weights_only
        checkpoint = torch.load(model_path, map_location="cpu")
    
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
    if benchmark_path.endswith(".zip"):
        print(f"Detected ZIP file: {benchmark_path}. Searching for .mat file inside...")
        with zipfile.ZipFile(benchmark_path, "r") as zip_ref:
            mat_files = [f for f in zip_ref.namelist() if f.endswith(".mat")]
            if not mat_files:
                raise FileNotFoundError("No .mat file found inside the ZIP archive.")
            
            # Use the first .mat file found
            mat_name = mat_files[0]
            target_path = os.path.join(os.path.dirname(benchmark_path), mat_name)
            
            if not os.path.exists(target_path):
                print(f"Extracting {mat_name} to {target_path}...")
                zip_ref.extract(mat_name, path=os.path.dirname(benchmark_path))
            else:
                print(f"Extracted file {mat_name} already exists.")
            
            benchmark_path = target_path

    print(f"Loading benchmark file: {benchmark_path}")
    try:
        mat_data = scipy.io.loadmat(benchmark_path)
        # The variable inside SIDD Benchmark MAT is 'BenchmarkNoisyBlocksSrgb'
        noisy_blocks = mat_data["BenchmarkNoisyBlocksSrgb"]  # Shape: (40, 32, 256, 256, 3)
    except NotImplementedError:
        print("Detected MATLAB v7.3 file. Using h5py for loading...")
        try:
            import h5py
        except ImportError:
            raise ImportError(
                "h5py is required to load MATLAB v7.3 files. "
                "Please install it with 'pip install h5py'."
            )
        
        with h5py.File(benchmark_path, "r") as f:
            # HDF5 (h5py) loads data in (C, W, H, blocks, scenes) order for v7.3 MAT files.
            # We must transpose to (scenes, blocks, H, W, C).
            dataset = f["BenchmarkNoisyBlocksSrgb"]
            noisy_blocks = np.array(dataset).transpose(4, 3, 2, 1, 0)
    
    num_scenes, num_blocks, H, W, C = noisy_blocks.shape
    print(f"Found {num_scenes} scenes with {num_blocks} blocks each ({H}x{W}px)")

    # 3. Initialize Inference Engine
    engine = HASSTInferenceEngine(model, device=device)

    # 4. Process Blocks
    denoised_blocks = np.zeros_like(noisy_blocks, dtype=np.uint8)
    output_blocks_base64string = []

    scene_pbar = tqdm(range(num_scenes), desc="Total Progress", mininterval=5.0)
    for s in scene_pbar:
        scene_pbar.set_description(f"Scene {s+1}/{num_scenes}")
        
        # --- NEW: Hybrid Scene-Level Stability ---
        # 1. Collect priors to extract the "Read Noise" (constant sensor noise)
        scene_read_vals = []
        for b in range(num_blocks):
            block = noisy_blocks[s, b].astype(np.float32) / 255.0
            block_tensor = torch.from_numpy(block).permute(2, 0, 1).unsqueeze(0).to(device)
            with torch.no_grad():
                # Channel 0: Shot (signal-dependent), Channel 1: Read (constant)
                prior = model.estimate_noise_prior(block_tensor)
                scene_read_vals.append(prior[:, 1:2].mean()) # Average spatially per block
        
        # 2. Compute a single stable Read Noise scalar for the entire scene
        avg_read_constant = torch.stack(scene_read_vals).mean()
        # ------------------------------------------

        for b in tqdm(range(num_blocks), desc=f"Scene {s+1}", leave=False, mininterval=1.0):
            # Preprocess: (H, W, C) [0, 255] -> (1, C, H, W) [0, 1]
            block = noisy_blocks[s, b].astype(np.float32) / 255.0
            block_tensor = torch.from_numpy(block).permute(2, 0, 1).unsqueeze(0).to(device)

            with torch.no_grad():
                # Get the block's specific prior (containing signal-dependent shot noise)
                block_prior = model.estimate_noise_prior(block_tensor)
                # HYBRID: Keep block-specific Shot noise, but inject scene-stable Read noise
                hybrid_prior = block_prior.clone()
                hybrid_prior[:, 1:2, ...] = avg_read_constant

            # Inference with mixed precision for maximum A100 throughput
            dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float16
            with torch.no_grad(), torch.autocast(device_type=device.type, dtype=dtype, enabled=device.type == "cuda"):
                if use_tta:
                    # 8x Geometric Self-Ensemble with TLC wrapper
                    pred_tensor = engine.forward_tlc(block_tensor, patch_size=patch_size, noise_prior=hybrid_prior)
                else:
                    # Single Forward Pass
                    pred_tensor = model(block_tensor, noise_prior=hybrid_prior)

            # Postprocess: (1, C, H, W) [0, 1] -> (H, W, C) [0, 255]
            pred_np = pred_tensor.squeeze(0).permute(1, 2, 0).float().cpu().numpy()
            out_block = (pred_np * 255.0).round().clip(0, 255).astype(np.uint8)
            
            denoised_blocks[s, b] = out_block
            output_blocks_base64string.append(array_to_base64string(out_block))

    # 5. Save Results
    print(f"Saving results to {output_path}...")

    # Save the original .mat file (required by standard SIDD script)
    mat_path = output_path.replace(".csv", ".mat")
    # We save both 'Idenoised' (Kaggle standard) and 'DenoisedBlocksSrgb' (SIDD standard)
    scipy.io.savemat(mat_path, {
        "Idenoised": denoised_blocks,
        "DenoisedBlocksSrgb": denoised_blocks
    })

    # Generate Kaggle expected CSV mapping format
    if output_path.endswith(".csv"):
        with open(output_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["ID", "BLOCK"])
            for i, block_b64 in enumerate(output_blocks_base64string):
                writer.writerow([i, block_b64])
        print(f"CSV payload generated at {output_path}")

    print("Prediction complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict SIDD Benchmark Blocks")
    parser.add_argument("--model", type=str, required=True, help="Path to best_model.pth")
    parser.add_argument("--benchmark", type=str, required=True, help="Path to BenchmarkNoisyBlocksSrgb.mat")
    parser.add_argument("--output", type=str, default="SubmitSrgb.csv", help="Output filename")
    parser.add_argument("--patch_size", type=int, default=256, help="Patch size for inference (use 256 for SIDD Benchmark)")
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
        patch_size=args.patch_size,
        allow_missing_mamba=args.allow_missing_mamba
    )
