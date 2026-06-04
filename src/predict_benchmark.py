import argparse
import base64
import os
import shutil
import tempfile
import zipfile

import h5py
import numpy as np
import pandas as pd
import scipy.io
import torch
from tqdm import tqdm

from models.hasst import HASST
from models.inference import HASSTInferenceEngine


def array_to_base64string(x):
    """Encodes a numpy array into a base64 string."""
    array_bytes = x.tobytes()
    base64_bytes = base64.b64encode(array_bytes)
    base64_string = base64_bytes.decode('utf-8')
    return base64_string


def predict_benchmark(model_path, benchmark_path, output_path, use_tta=True):
    """
    Inference script for the official SIDD Benchmark (sRGB).
    Loads BenchmarkNoisyBlocksSrgb.mat (or .zip) and saves a Base64 encoded CSV.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load Model
    print(f"Loading checkpoint from {model_path}...")
    try:
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    except TypeError:
        # Fallback for older torch versions that don't support weights_only
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
    temp_dir = None
    actual_benchmark_path = benchmark_path

    if benchmark_path.endswith(".zip"):
        print(f"Detected ZIP file: {benchmark_path}. Extracting...")
        temp_dir = tempfile.mkdtemp()
        with zipfile.ZipFile(benchmark_path, 'r') as z:
            mat_files = [f for f in z.namelist() if f.endswith(".mat")]
            if not mat_files:
                raise FileNotFoundError("No .mat file found inside the ZIP archive.")
            # Use the first .mat file found
            z.extract(mat_files[0], temp_dir)
            actual_benchmark_path = os.path.join(temp_dir, mat_files[0])
            print(f"Extracted to {actual_benchmark_path}")

    print(f"Loading benchmark file: {actual_benchmark_path}")
    key = "BenchmarkNoisyBlocksSrgb"
    
    try:
        try:
            mat_data = scipy.io.loadmat(actual_benchmark_path)
            noisy_blocks = mat_data[key]
        except NotImplementedError:
            print("Detected MATLAB v7.3 file. Loading with h5py...")
            with h5py.File(actual_benchmark_path, 'r') as f:
                # h5py loads with shape (C, W, H, num_blocks, num_scenes)
                # We need (num_scenes, num_blocks, H, W, C)
                noisy_blocks = np.array(f[key])
                noisy_blocks = noisy_blocks.transpose(4, 3, 2, 1, 0)
    finally:
        # Cleanup temporary directory if it was created
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            print("Cleaned up temporary extraction directory.")
    
    # Shape: (40, 32, 256, 256, 3)
    num_scenes, num_blocks, H, W, C = noisy_blocks.shape
    print(f"Found {num_scenes} scenes with {num_blocks} blocks each ({H}x{W}px)")

    # 3. Initialize Inference Engine
    engine = HASSTInferenceEngine(model, device=device)

    # 4. Process Blocks
    denoised_blocks = np.zeros_like(noisy_blocks, dtype=np.uint8)

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
            pred_np = pred_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
            denoised_blocks[s, b] = (pred_np * 255.0).round().astype(np.uint8)

    # 5. Save Results
    print(f"Saving results to {output_path}...")
    
    output_blocks_base64string = []
    for s in range(num_scenes):
        for b in range(num_blocks):
            # Each block is (H, W, C) uint8
            out_block = denoised_blocks[s, b]
            out_block_base64string = array_to_base64string(out_block)
            output_blocks_base64string.append(out_block_base64string)

    output_df = pd.DataFrame()
    n_total_blocks = len(output_blocks_base64string)
    print(f"Total number of blocks to save: {n_total_blocks}")
    
    output_df['ID'] = np.arange(n_total_blocks)
    output_df['BLOCK'] = output_blocks_base64string
    
    output_df.to_csv(output_path, index=False)
    print(f"Prediction complete! CSV saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict SIDD Benchmark Blocks")
    parser.add_argument("--model", type=str, required=True, help="Path to best_model.pth")
    parser.add_argument("--benchmark", type=str, required=True, help="Path to BenchmarkNoisyBlocksSrgb.mat")
    parser.add_argument("--output", type=str, default="SubmitSrgb.csv", help="Output filename")
    parser.add_argument("--no_tta", action="store_true", help="Disable Test-Time Augmentation (faster but lower PSNR)")
    
    args = parser.parse_args()
    
    # Ensure src is in PYTHONPATH if running from root
    predict_benchmark(args.model, args.benchmark, args.output, use_tta=not args.no_tta)
