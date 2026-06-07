# Hybrid Attentive State-Space Transformer (HASST) for SIDD sRGB Denoising

This repository implements the **HASST** framework, a hybrid architecture designed to achieve state-of-the-art results in the SIDD sRGB Denoising Benchmark. It combines local feature extraction, global state-space modeling, and noise-aware modulation.

## Visual Comparison

|        Noisy Input         |          HASST Denoised          |          Ground Truth          |
|:--------------------------:|:--------------------------------:|:------------------------------:|
| ![Noisy](assets/noisy.PNG) | ![Denoised](assets/restored.PNG) | ![GT](assets/ground_truth.PNG) |

## Project Structure

```text
.
├── configs/            # YAML configuration files for experiments
├── data/               # Benchmark data (e.g., BenchmarkNoisyBlocksSrgb.mat)
├── dataset/            # Training datasets (e.g., SIDD Medium LMDB)
├── src/                # Core source code
│   ├── data/           # Dataset loaders and augmentations
│   ├── losses/         # Custom loss functions (Charbonnier, Wavelet)
│   ├── models/         # HASST, NAFNet, and MambaIR architectures
│   ├── utils/          # Metrics (PSNR/SSIM) and logging
│   ├── config.py       # Configuration system
│   ├── train.py        # Main training script
│   └── predict_benchmark.py # Inference script for SIDD Benchmark
├── tools/              # Utility scripts for analysis and environment checks
└── checkpoints/        # Saved model weights (.pth files)
```

## Technical Pillars

### 1. Hybrid Architecture
HASST bridges the gap between local and global modeling:
- **Local Details**: [NAFNet](https://github.com/megvii-research/NAFNet)-based blocks capture high-frequency details using a Nonlinear Activation Free approach.
- **Global Context**: [MambaIRv2](https://github.com/csguoh/MambaIR)-based Attentive State Space blocks model long-range dependencies with linear complexity.
- **Noise Awareness**: The [LoNPE](https://github.com/BNU-ERC-ITEA/Condformer) module estimates spatial noise maps to modulate features dynamically.

### 2. Progressive Learning & Augmentation
Training automatically scales from small patches (128x128) to high-resolution ones (384x384). This "coarse-to-fine" strategy is complemented by frequency-domain augmentations to improve robustness against various noise distributions.

### 3. Kaggle-Native Resiliency
The pipeline is optimized for constrained 12-hour GPU sessions:
- **Time-Aware Exit**: Gracefully shuts down at 11.5 hours, saving the exact optimizer and scheduler states.
- **Auto-Resume**: Automatically detects the latest checkpoint in the output directory to continue training seamlessly.
- **Efficiency**: Utilizes `torch.compile` and mixed-precision (FP16/BF16/FP32) to maximize GPU throughput.

## Getting Started

### Installation
This project uses `uv` for dependency management.

```bash
# Core setup (Windows/Generic)
uv sync

# Full setup (Linux/CUDA with Mamba support)
uv sync --extra mamba
```

### Usage Workflow

1. **Prepare Data**: Convert SIDD PNGs to LMDB for high-speed I/O.
   ```bash
   uv run python src/create_lmdb.py --data_dir /path/to/sidd/Data --lmdb_dir dataset/sidd_lmdb
   ```

2. **Train**: Launch with `accelerate`.
   ```bash
   uv run accelerate launch src/train.py --config configs/train_config_example.yaml
   ```

3. **Inference**: Generate a Kaggle-ready submission from the benchmark file.
   ```bash
   uv run python src/predict_benchmark.py \
       --model checkpoints/best_model.pth \
       --benchmark data/BenchmarkNoisyBlocksSrgb.mat \
       --output results/submission.csv
   ```

## Configuration
All parameters are managed via YAML. See `configs/train_config_example.yaml` for a fully documented template covering model dimensions, training schedules, and loss weights.
