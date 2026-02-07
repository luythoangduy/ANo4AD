# RDPP Noising Experiments Guide

## Overview

This guide covers running comprehensive experiments for RDPP Noising with different configurations:
- **Noise Types**: No noise, Uniform, Gaussian, Perlin
- **Noise Positions**: After Encoder, After Projector, After MFF_OCE
- **Sampling Methods**: Greedy, Random, K-means

## Prerequisites

### Quick Setup (Recommended)

#### Linux/Mac:
```bash
chmod +x setup.sh
./setup.sh
```

#### Windows (PowerShell):
```powershell
powershell -ExecutionPolicy Bypass -File setup.ps1
```

The setup script will automatically:
- Install all required packages (including perlin-numpy)
- Download MVTec AD dataset from Google Drive
- Extract and organize dataset files
- Generate benchmark metadata
- Verify installation

### Manual Installation

If you prefer manual setup:

```bash
pip install perlin-numpy gdown
pip install adeval FrEIA geomloss ninja faiss-cpu einops numba imgaug scikit-image opencv-python fvcore tensorboardX timm
```

Then download and prepare the dataset manually following the instructions in [data/README.md](data/README.md).

### Make Scripts Executable (Linux/Mac only)

```bash
chmod +x run_all_rdpp_experiments.sh
chmod +x run_rdpp_single.sh
chmod +x run_rdpp_by_group.sh
```

**Windows users**: Use `.bat` files or run Python commands directly (see Windows section below).

### Weights & Biases (wandb) Setup (Recommended)

wandb is **enabled by default** for cloud-based experiment tracking.

#### Setup wandb:

**Linux/Mac:**
```bash
chmod +x wandb_setup.sh
./wandb_setup.sh
```

**Windows:**
```powershell
powershell -ExecutionPolicy Bypass -File wandb_setup.ps1
```

**Manual setup:**
```bash
pip install wandb
wandb login
```

#### Key Features:
- ✅ Automatic cloud backup of all experiments
- ✅ Interactive visualizations and comparisons
- ✅ Model checkpoint versioning
- ✅ Team collaboration
- ✅ Free for academic research

**Note**: All experiments will automatically sync to your wandb account. To disable wandb for a run:
```bash
python run.py -c ... wandb.enable=False
```

For detailed wandb guide, see [WANDB_GUIDE.md](WANDB_GUIDE.md).

## Experiment Structure

### Total Experiments: 28

1. **No Noise Baseline**: 1 experiment
2. **With Noise**: 27 experiments (3 noise types × 3 positions × 3 sampling methods)

### Noise Types (3)
- `uniform`: Uniform random noise in [-std, +std]
- `gaussian`: Gaussian noise with N(0, std²)
- `perlin`: Perlin noise (smooth, natural-looking noise)

### Noise Positions (3)
- `encoder`: Apply noise after encoder features (feats_t)
- `projector`: Apply noise after projection layer
- `mff_oce`: Apply noise after MFF_OCE module

### Sampling Methods (3)
- `greedy`: Greedy k-center coreset sampling (default, best diversity)
- `random`: Fast random sampling
- `kmeans`: K-means clustering centroids

## Running Experiments

### 1. Run All Experiments (28 total)

```bash
./run_all_rdpp_experiments.sh
```

This will:
- Run all 28 experiment combinations
- Save logs for each experiment
- Generate a summary report
- Take approximately 28 × training_time

**Estimated time**: ~28 hours (assuming 1 hour per experiment with 100 epochs)

### 2. Run Single Experiment

```bash
./run_rdpp_single.sh <noise_type> <noise_position> <sampling_method>
```

Examples:
```bash
# No noise baseline
./run_rdpp_single.sh none none none

# Uniform noise after encoder with greedy sampling
./run_rdpp_single.sh uniform encoder greedy

# Gaussian noise after projector with random sampling
./run_rdpp_single.sh gaussian projector random

# Perlin noise after mff_oce with k-means sampling
./run_rdpp_single.sh perlin mff_oce kmeans
```

### 3. Run Grouped Experiments

Compare specific aspects by running grouped experiments:

#### Compare Noise Types (3 experiments)
```bash
./run_rdpp_by_group.sh noise_types
```
Runs: uniform, gaussian, perlin (all with encoder + greedy)

#### Compare Noise Positions (3 experiments)
```bash
./run_rdpp_by_group.sh positions
```
Runs: encoder, projector, mff_oce (all with uniform + greedy)

#### Compare Sampling Methods (3 experiments)
```bash
./run_rdpp_by_group.sh sampling
```
Runs: greedy, random, kmeans (all with uniform + encoder)

#### Baseline Comparison (2 experiments)
```bash
./run_rdpp_by_group.sh baseline
```
Runs: no_noise baseline + uniform_encoder_greedy

## Experiment Naming Convention

Experiments are named as: `{noise_type}_{noise_position}_{sampling_method}`

Examples:
- `no_noise_baseline`
- `uniform_encoder_greedy`
- `gaussian_projector_random`
- `perlin_mff_oce_kmeans`

## Output and Logs

### Directory Structure
```
results/rdpp_experiments_YYYYMMDD_HHMMSS/
├── experiments.log              # Main log with all experiments
├── exp_1_no_noise.log          # Individual experiment logs
├── exp_2_uniform_encoder_greedy.log
├── exp_3_uniform_encoder_random.log
└── ...
```

### Tensorboard Logs
Training logs are saved in the logdir specified in config:
```
logs/{experiment_name}/
```

## Configuration Details

### Base Configuration
File: `configs/rdpp_noising/rdpp_noising_256_100e.py`

Key parameters:
- **Epochs**: 100
- **Image size**: 256×256
- **Batch size**: 16
- **Learning rate**: 0.005
- **Coreset sampling ratio**: 0.01 (1% of features)
- **Noise std range**: (0.01, 0.3)
- **N neighbors**: 9

### Override Parameters

You can override any config parameter via command line:

```bash
python run.py -c configs/rdpp_noising/rdpp_noising_256_100e.py -m train \
    model.kwargs.enable_noise=True \
    model.kwargs.noise_type="gaussian" \
    model.kwargs.noise_position="projector" \
    model.kwargs.noise_std_range="(0.01, 0.5)" \
    trainer.sampling_method="kmeans" \
    trainer.epoch_full=50
```

## Quick Start Guide

### For Quick Testing (Recommended First Steps)

1. **Test baseline** (no noise):
```bash
./run_rdpp_single.sh none none none
```

2. **Test one configuration** (uniform + encoder + greedy):
```bash
./run_rdpp_single.sh uniform encoder greedy
```

3. **Compare noise types** (3 experiments):
```bash
./run_rdpp_by_group.sh noise_types
```

### For Full Experiments

Run all 28 experiments:
```bash
./run_all_rdpp_experiments.sh
```

## Monitoring Progress

### Check running experiment
```bash
tail -f results/rdpp_experiments_*/experiments.log
```

### Monitor specific experiment
```bash
tail -f results/rdpp_experiments_*/exp_*.log
```

### View tensorboard
```bash
tensorboard --logdir logs/
```

## Expected Outcomes

### What to Compare

1. **Noise Types**: Which noise distribution works best?
   - Uniform: Simple, fast
   - Gaussian: Natural, well-studied
   - Perlin: Smooth, spatially coherent

2. **Noise Positions**: Where to inject noise?
   - Encoder: Early injection, affects all downstream
   - Projector: After projection, before fusion
   - MFF_OCE: Late injection, after fusion

3. **Sampling Methods**: How to build memory bank?
   - Greedy: Best coverage, slower
   - Random: Fast, less optimal
   - K-means: Good clustering, moderate speed

### Evaluation Metrics

Key metrics to compare:
- `mAUROC_sp_max`: Sample-level AUROC (anomaly detection)
- `mAUROC_px`: Pixel-level AUROC (anomaly localization)
- `mAUPRO_px`: Pixel-level AUPRO
- `mF1_max_px`: Max F1 score for pixel-level

## Troubleshooting

### Out of Memory
Reduce batch size:
```bash
trainer.data.batch_size=8
```

### Perlin Noise Not Available
If you see warnings about Perlin noise, install:
```bash
pip install perlin-numpy
```

### Slow K-means
For large datasets, k-means may be slow. Consider:
- Using greedy or random sampling instead
- Reducing coreset_sampling_ratio

## Advanced Usage

### Custom Noise Parameters

Adjust noise intensity:
```bash
model.kwargs.noise_std_range="(0.05, 0.8)"  # Stronger noise
```

### Different Sampling Ratios

```bash
model.kwargs.coreset_sampling_ratio=0.02  # Use 2% of features
```

### Multiple GPUs

Edit GPU_ID in the script or:
```bash
CUDA_VISIBLE_DEVICES=0,1 python run.py ...
```

## Windows-Specific Instructions

### Running on Windows

Windows users can use the provided `.bat` file or run Python commands directly.

#### Using Batch File

```cmd
run_rdpp_single.bat uniform encoder greedy
run_rdpp_single.bat gaussian projector random
run_rdpp_single.bat none none none
```

#### Using Python Directly

```cmd
REM No noise baseline
python run.py -c configs/rdpp_noising/rdpp_noising_256_100e.py -m train model.kwargs.enable_noise=False trainer.logdir_sub="no_noise_baseline"

REM Uniform + encoder + greedy
python run.py -c configs/rdpp_noising/rdpp_noising_256_100e.py -m train model.kwargs.enable_noise=True model.kwargs.noise_type="uniform" model.kwargs.noise_position="encoder" trainer.sampling_method="greedy" trainer.logdir_sub="uniform_encoder_greedy"

REM Gaussian + projector + random
python run.py -c configs/rdpp_noising/rdpp_noising_256_100e.py -m train model.kwargs.enable_noise=True model.kwargs.noise_type="gaussian" model.kwargs.noise_position="projector" trainer.sampling_method="random" trainer.logdir_sub="gaussian_projector_random"

REM Perlin + mff_oce + kmeans
python run.py -c configs/rdpp_noising/rdpp_noising_256_100e.py -m train model.kwargs.enable_noise=True model.kwargs.noise_type="perlin" model.kwargs.noise_position="mff_oce" trainer.sampling_method="kmeans" trainer.logdir_sub="perlin_mff_oce_kmeans"
```

#### Running All Experiments on Windows

Create a batch file or run experiments one by one:

```cmd
@echo off
for %%t in (uniform gaussian perlin) do (
    for %%p in (encoder projector mff_oce) do (
        for %%s in (greedy random kmeans) do (
            echo Running %%t %%p %%s
            python run.py -c configs/rdpp_noising/rdpp_noising_256_100e.py -m train model.kwargs.enable_noise=True model.kwargs.noise_type="%%t" model.kwargs.noise_position="%%p" trainer.sampling_method="%%s" trainer.logdir_sub="%%t_%%p_%%s"
        )
    )
)
```

## Summary

This experiment suite provides comprehensive evaluation of RDPP Noising across:
- 4 noise configurations (none + 3 types)
- 3 noise positions
- 3 sampling strategies

### Key Files Created

- **[setup.sh](setup.sh)** / **[setup.ps1](setup.ps1)**: Automated setup for Linux/Mac and Windows
- **[run_all_rdpp_experiments.sh](run_all_rdpp_experiments.sh)**: Run all 28 experiments (Linux/Mac)
- **[run_rdpp_single.sh](run_rdpp_single.sh)** / **[run_rdpp_single.bat](run_rdpp_single.bat)**: Run single experiment
- **[run_rdpp_by_group.sh](run_rdpp_by_group.sh)**: Run grouped experiments (Linux/Mac)
- **[model/rdpp_noising.py](model/rdpp_noising.py)**: Updated model with noise types, positions, and sampling methods

Use the provided scripts to run experiments efficiently and systematically compare results.

For questions or issues, check the logs in `results/rdpp_experiments_*/` directory.
