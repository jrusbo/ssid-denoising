# Hybrid Attentive State-Space Transformer (HASST) for SIDD sRGB Denoising

This repository contains the official codebase for the SIDD sRGB Denoising Benchmark. By leveraging a progressive training schedule, frequency-based augmentations, and a novel hybrid architecture.
Get the dataset from the [SIDD website](https://abdokamel.github.io/sidd/). Download the SIDD Medium dataset, sRGB images only (~12 GB) and the [BenchmarkNoisyBlocksSrgb.mat](https://www.kaggle.com/competitions/sidd-benchmark-srgb-psnr/data) for inference.

## How It Works

The HASST framework is built on several key technical pillars designed specifically for the unique challenges of sRGB denoising:

### 1. Hybrid Architectural Design
Standard CNNs are great for local textures but fail at long-range dependencies. Transformers capture global context but are computationally expensive. HASST uses a verified hybrid approach:
- **Local Detail Extraction:** [NAFNet](https://github.com/megvii-research/NAFNet)-based blocks capture high-frequency details using a Nonlinear Activation Free approach (SimpleGate + SCA).
- **Global Modeling:** [MambaIRv2](https://github.com/csguoh/MambaIR)-based Attentive State Space blocks capture global scene structures with linear complexity via non-causal selective scans.
- **Noise Priors:** The [LoNPE](https://github.com/BNU-ERC-ITEA/Condformer) (Locally Noise Prior Estimation) module from Condformer estimates a dense spatial noise map to modulate features according to local noise intensity.

### 2. Progressive Learning Schedule
Training starts with small patches (128x128) and large batch sizes for fast convergence and global feature learning. It then scales up to larger patches (up to 384x384) to refine local details and handle high-resolution structures.

### 3. Optimization & Resiliency (Kaggle Optimized)
- **Time-Aware Training:** The script monitors execution time and performs a graceful exit with a final checkpoint at 11.5 hours, just before Kaggle's 12-hour limit.
- **Exact State Resuming:** Resuming restores the model, optimizer, and scheduler states perfectly, ensuring no training discontinuities.
- **Dual Checkpointing:** Periodic checkpoints are saved locally for speed and uploaded to WandB as Artifacts (Best & Last models) for remote safety.
- **`torch.compile`:** Uses PyTorch 2.x kernel fusion to maximize GPU throughput.

## Setup & Usage

### 1. Installation
The project relies on `uv` for dependency management. Due to architectural dependencies on `triton`, some components (Mamba) are only available on Linux/CUDA environments.

**Standard Setup (Windows / Generic):**
Installs the core framework (NAFNet, LoNPE, Training Pipeline). Mamba blocks will fallback to Identity.
```bash
# Install core dependencies
uv sync
```

**Full Setup (Linux / Kaggle / CUDA):**
Installs the full HASST architecture including global State-Space Modeling.
```bash
# Install core + mamba extensions
uv sync --extra mamba
```

**Activate environment (optional):**
```bash
source .venv/bin/activate
```

---

### 2. Multi-Config Execution
HASST now supports processing multiple experiments in sequence. You can point the training script to a single YAML file or an entire directory of configurations.

**Run a single experiment:**
```bash
# Point to your YAML file
uv run accelerate launch src/train.py --config configs/train_config.yaml
```

**Batch process multiple experiments:**
Place your YAML files in a folder (e.g., `configs/experiments/`) and the script will process them one by one, skipping failing ones with a log.
```bash
uv run accelerate launch src/train.py --config configs/
```

### 3. Configuration System
Every aspect of HASST is controlled via YAML. We provide `configs/train_config_example.yaml` as a comprehensive template with 25+ documented variables.

**How to Configure:**
1. **Copy the example:** `cp configs/train_config_example.yaml configs/my_run.yaml`
2. **Tweak Parameters:** Adjust `embed_dim` for model size, `total_iters` for duration, `loss_weights` for tuning...
3. **Execute:** Pass your config to the training script.

### 4. Practical Execution Guide (Step-by-Step)

#### Step A: Data Ingestion (LMDB)
```bash
# Standard PNG files are too slow for high-end GPUs. This script converts them.
uv run python src/create_lmdb.py --data_dir /path/to/sidd/Data --lmdb_dir /kaggle/working/sidd_lmdb
```

#### Step B: Training with `accelerate`
```bash
# First time setup (interactive)
uv run accelerate config

# Start training with a specific config
uv run accelerate launch src/train.py --config configs/default.yaml
```

#### Step C: High-Quality Inference (Benchmark)
```bash
# Generate the final submission for the SIDD Benchmark
uv run python src/predict_benchmark.py --model checkpoints/best_model.pth --benchmark /path/to/BenchmarkNoisyBlocksSrgb.mat --output SubmitSrgb.mat
```

---

## Deep Dive: Kaggle Resiliency

Training on Kaggle presents unique challenges like the **12-hour session limit** and potential internet flakes. HASST is built to be "Kaggle-Native":

- **The 12-Hour Shield:** Set `max_hours: 11.5` in your config. The script monitors its own age. At 11.5 hours, it performs a high-priority "Safe Exit": it saves the model, optimizer, and scheduler states locally and shuts down gracefully before Kaggle kills the process.
- **Auto-Resume Logic:** When you restart your session, the script automatically scans `output_dir` for the latest `last_model.pth`. It restores the **exact state** (including optimizer momentum and learning rate) so your training curve remains perfectly smooth.
- **Performance-First Logging:** While metrics are streamed to WandB in real-time, model uploads are deferred to the end of the session. Only the "Best Model" (based on PSNR) and the "Final Session Model" are uploaded as WandB Artifacts, ensuring network latency doesn't slow down your training iterations.
- **Local persistence:** Kaggle environments are set to persist `/kaggle/working`. Our script prioritizes local saving for speed while using WandB as a redundant cloud backup for your most valuable results.

## Repository Structure

- `configs/`: YAML configurations. Use `train_config_example.yaml` as your starting point.
- `src/data/`: High-speed LMDB handling, creation scripts, and progressive augmentations.
- `src/models/`: The core HASST hybrid architecture, including LoNPE, NAFNet blocks, and Mamba-based global context.
- `src/losses/`: Dual-domain loss functions (Spatial Charbonnier + Frequency Wavelet).
- `src/utils/`: Telemetry, interactive visual logging, and validation metrics (PSNR/SSIM).