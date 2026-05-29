import argparse

import numpy as np
import scipy.io
import torch
from tqdm import tqdm

from models.hasst import HASST
from models.inference import HASSTInferenceEngine


def predict_benchmark(model_path, benchmark_path, output_path, use_tta=True):
    """
    Inference script for the official SIDD Benchmark (sRGB).
    Loads BenchmarkNoisyBlocksSrgb.mat and saves SubmitSrgb.mat.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load Model
    print(f"Loading checkpoint from {model_path}...")
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
    mat_data = scipy.io.loadmat(benchmark_path)
    # The variable inside SIDD Benchmark MAT is usually 'BenchmarkNoisyBlocksSrgb'
    noisy_blocks = mat_data["BenchmarkNoisyBlocksSrgb"]  # Shape: (40, 32, 256, 256, 3)
    
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
                    # 8x Geometric Self-Ensemble with TLC wrapper
                    pred_tensor = engine.forward_tlc(block_tensor, patch_size=256)
                else:
                    # Single Forward Pass
                    pred_tensor = model(block_tensor)

            # Postprocess: (1, C, H, W) [0, 1] -> (H, W, C) [0, 255]
            pred_np = pred_tensor.squeeze(0).permute(1, 2, 0).cpu().clamp(0.0, 1.0).numpy()
            denoised_blocks[s, b] = (pred_np * 255.0).round().astype(np.uint8)

    # 5. Save Results
    print(f"Saving results to {output_path}...")
    # Variable name MUST be 'DenoisedBlocksSrgb' for the Kaggle submission system
    scipy.io.savemat(output_path, {"DenoisedBlocksSrgb": denoised_blocks})
    print("Prediction complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict SIDD Benchmark Blocks")
    parser.add_argument("--model", type=str, required=True, help="Path to best_model.pth")
    parser.add_argument("--benchmark", type=str, required=True, help="Path to BenchmarkNoisyBlocksSrgb.mat")
    parser.add_argument("--output", type=str, default="SubmitSrgb.mat", help="Output filename")
    parser.add_argument("--no_tta", action="store_true", help="Disable Test-Time Augmentation (faster but lower PSNR)")
    
    args = parser.parse_args()
    
    # Ensure src is in PYTHONPATH if running from root
    predict_benchmark(args.model, args.benchmark, args.output, use_tta=not args.no_tta)
