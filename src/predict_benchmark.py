import argparse
import base64
import os
import tempfile
import zipfile
from pathlib import Path

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
    base64_string = base64_bytes.decode('utf-8')
    return base64_string


def predict_benchmark(model_path, benchmark_path, output_path, use_tta=True):
    """
    Inference script for the official SIDD Benchmark (sRGB).
    Loads BenchmarkNoisyBlocksSrgb.mat and saves SubmitSrgb.mat and SubmitSrgb.csv.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load Model
    print(f"Loading checkpoint from {model_path}...")
    try:
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    except TypeError:
        # Fallback for older PyTorch versions (< 2.4)
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
        
    model.load_state_dict(clean_state_dict)
    model = model.to(device)
    model.eval()

    # 2. Load Benchmark Data
    print(f"Loading benchmark file: {benchmark_path}")
    
    # Handle zipped benchmark file
    if str(benchmark_path).endswith(".zip"):
        print("Zip file detected. Extracting...")
        temp_dir = tempfile.TemporaryDirectory()
        with zipfile.ZipFile(benchmark_path, 'r') as zip_ref:
            # Look for a .mat file inside the zip
            mat_files = [f for f in zip_ref.namelist() if f.endswith(".mat")]
            if not mat_files:
                raise FileNotFoundError(f"No .mat file found inside {benchmark_path}")
            
            # Prefer 'BenchmarkNoisyBlocksSrgb.mat' if it exists, otherwise take the first one
            target_mat = "BenchmarkNoisyBlocksSrgb.mat" if "BenchmarkNoisyBlocksSrgb.mat" in mat_files else mat_files[0]
            zip_ref.extract(target_mat, temp_dir.name)
            benchmark_path = os.path.join(temp_dir.name, target_mat)
            print(f"Extracted {target_mat} to temporary directory.")

    mat_data = scipy.io.loadmat(benchmark_path)
    # The variable inside SIDD Benchmark MAT is usually 'BenchmarkNoisyBlocksSrgb'
    noisy_blocks = mat_data["BenchmarkNoisyBlocksSrgb"]  # Shape: (40, 32, 256, 256, 3)
    
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
        for b in tqdm(range(num_blocks), desc=f"Scene {s+1}", leave=False, mininterval=1.0):
            # Preprocess: (H, W, C) [0, 255] -> (1, C, H, W) [0, 1]
            block = noisy_blocks[s, b].astype(np.float32) / 255.0
            block_tensor = torch.from_numpy(block).permute(2, 0, 1).unsqueeze(0).to(device)

            # Inference with mixed precision for maximum A100 throughput
            # We use bfloat16 on A100 as it's faster and more stable than fp16
            dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float16
            with torch.no_grad(), torch.autocast(device_type=device.type, dtype=dtype, enabled=device.type == "cuda"):
                if use_tta:
                    # 8x Geometric Self-Ensemble with TLC wrapper
                    pred_tensor = engine.forward_tlc(block_tensor, patch_size=256)
                else:
                    # Single Forward Pass
                    pred_tensor = model(block_tensor)

            # Postprocess: (1, C, H, W) [0, 1] -> (H, W, C) [0, 255]
            # NumPy does not always support bfloat16 tensors from autocast inference.
            pred_np = pred_tensor.squeeze(0).permute(1, 2, 0).float().cpu().numpy()
            denoised_block = (pred_np * 255.0).round().astype(np.uint8)
            denoised_blocks[s, b] = denoised_block
            
            # Convert to base64 for Kaggle submission
            output_blocks_base64string.append(array_to_base64string(denoised_block))

    # 5. Save Results
    print(f"Saving results to {output_path}...")

    # Save the original .mat file (required by standard SIDD script)
    mat_path = output_path.replace(".csv", ".mat")
    scipy.io.savemat(mat_path, {"DenoisedBlocksSrgb": denoised_blocks})

    # Generate Kaggle expected CSV mapping format
    if output_path.endswith(".csv"):
        import csv
        with open(output_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["ID", "BLOCK"])
            for idx, b64_str in enumerate(output_blocks_base64string):
                writer.writerow([idx, b64_str])
        print(f"CSV payload generated at {output_path}")

    print("Prediction complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict SIDD Benchmark Blocks")
    parser.add_argument("--model", type=str, required=True, help="Path to best_model.pth")
    parser.add_argument("--benchmark", type=str, required=True, help="Path to BenchmarkNoisyBlocksSrgb.mat")
    parser.add_argument("--output", type=str, default="SubmitSrgb.csv", help="Output filename")
    parser.add_argument("--no_tta", action="store_true", help="Disable Test-Time Augmentation (faster but lower PSNR)")
    
    args = parser.parse_args()
    
    predict_benchmark(args.model, args.benchmark, args.output, use_tta=not args.no_tta)

