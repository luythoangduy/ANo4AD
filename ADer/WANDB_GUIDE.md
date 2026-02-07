# Weights & Biases Integration Guide

## Overview

This guide covers using Weights & Biases (wandb) for tracking RDPP Noising experiments instead of local logging.

### Benefits of Using wandb

- ✅ **Cloud Storage**: All experiments automatically synced to cloud
- ✅ **Visualization**: Interactive charts, metrics, and comparisons
- ✅ **Collaboration**: Share experiments with team members
- ✅ **Model Versioning**: Track and version model checkpoints
- ✅ **Experiment Comparison**: Compare multiple runs side-by-side
- ✅ **Free for Academics**: Free unlimited runs for academic research

---

## Quick Setup

### 1. Install wandb

```bash
pip install wandb
```

### 2. Run Setup Script

#### Linux/Mac:
```bash
chmod +x wandb_setup.sh
./wandb_setup.sh
```

#### Windows (PowerShell):
```powershell
powershell -ExecutionPolicy Bypass -File wandb_setup.ps1
```

The setup script will:
1. Install wandb
2. Log you in to your wandb account
3. Configure wandb settings
4. Test the connection

### 3. Get Your API Key

If you don't have a wandb account yet:

1. Go to [https://wandb.ai/](https://wandb.ai/)
2. Sign up (free for academics)
3. Get your API key from [https://wandb.ai/authorize](https://wandb.ai/authorize)

---

## Configuration

### Default Configuration

The RDPP Noising config already has wandb enabled by default:

```python
# In configs/rdpp_noising/rdpp_noising_256_100e.py
self.wandb = Namespace()
self.wandb.enable = True  # wandb is ON by default
self.wandb.project = 'rdpp-noising-experiments'
self.wandb.entity = None  # Your wandb username/team
self.wandb.name = None  # Auto-generated from experiment
self.wandb.tags = ['rdpp', 'noising', '100epochs']
self.wandb.notes = 'RDPP Noising experiments with adaptive noise injection'
self.wandb.log_model = True  # Save best checkpoints to wandb
self.wandb.log_freq = 50  # Log every 50 iterations
```

### Customize wandb Settings

#### Set Your Username/Team

Edit `configs/rdpp_noising/rdpp_noising_256_100e.py`:

```python
self.wandb.entity = "your-username"  # or "your-team-name"
```

#### Change Project Name

```python
self.wandb.project = "my-custom-project"
```

#### Disable Model Checkpoints (Save Space)

```python
self.wandb.log_model = False
```

---

## Running Experiments with wandb

### 1. Using Shell Scripts (Automatic wandb Integration)

The experiment scripts automatically set wandb run names and tags:

```bash
# Runs will be named automatically based on config
./run_rdpp_single.sh uniform encoder greedy
# → wandb run name: "uniform_encoder_greedy"
# → tags: ['rdpp', 'noising', 'uniform', 'encoder', 'greedy']

./run_rdpp_single.sh none none none
# → wandb run name: "no_noise_baseline"
# → tags: ['rdpp', 'noising', 'no-noise', 'baseline']
```

### 2. Using Python Directly

```bash
python run.py \
    -c configs/rdpp_noising/rdpp_noising_256_100e.py \
    -m train \
    model.kwargs.enable_noise=True \
    model.kwargs.noise_type="gaussian" \
    model.kwargs.noise_position="projector" \
    trainer.sampling_method="random" \
    wandb.name="my_custom_run_name" \
    wandb.tags="['custom','tag1','tag2']"
```

### 3. Disable wandb for Specific Run

```bash
# Disable wandb for this run only
python run.py -c configs/rdpp_noising/rdpp_noising_256_100e.py -m train wandb.enable=False

# Or with shell script
CUDA_VISIBLE_DEVICES=0 python run.py -c configs/rdpp_noising/rdpp_noising_256_100e.py -m train wandb.enable=False
```

---

## wandb Modes

### Online Mode (Default)

Experiments sync to cloud in real-time:

```bash
python run.py -c configs/rdpp_noising/rdpp_noising_256_100e.py -m train
```

### Offline Mode

Run experiments offline, sync later:

```bash
# Linux/Mac
export WANDB_MODE=offline
python run.py -c configs/rdpp_noising/rdpp_noising_256_100e.py -m train

# Windows (PowerShell)
$env:WANDB_MODE='offline'
python run.py -c configs/rdpp_noising/rdpp_noising_256_100e.py -m train

# Sync later
wandb sync wandb/offline-run-YYYYMMDD_HHMMSS-XXXXX
```

### Disabled Mode

Completely disable wandb (no logging at all):

```bash
# Linux/Mac
export WANDB_MODE=disabled

# Windows
$env:WANDB_MODE='disabled'
```

---

## Viewing Experiments

### Web Dashboard

1. Go to [https://wandb.ai/](https://wandb.ai/)
2. Navigate to your project: `https://wandb.ai/YOUR_USERNAME/rdpp-noising-experiments`
3. View all experiments, charts, and comparisons

### Command Line

```bash
# Open current run in browser
wandb online

# List all runs
wandb runs YOUR_USERNAME/rdpp-noising-experiments

# View run summary
wandb summary
```

---

## What Gets Logged to wandb

### Automatically Logged

1. **Training Metrics** (every 50 iterations):
   - Loss values (cos, proj)
   - Learning rate
   - Batch time, data loading time
   - Influence statistics
   - Noise std statistics

2. **Test Metrics** (per epoch):
   - mAUROC (sample-level and pixel-level)
   - mAP, mF1, mAUPRO
   - mIoU, mAcc

3. **System Metrics**:
   - GPU usage
   - CPU usage
   - Memory usage
   - Disk I/O

4. **Configuration**:
   - All hyperparameters
   - Model architecture
   - Data augmentations
   - Noise configuration

5. **Model Checkpoints** (if `log_model=True`):
   - Best model checkpoint
   - Final model checkpoint

### Custom Logging

To log additional metrics in your trainer code:

```python
import wandb

# Log custom metrics
wandb.log({
    "custom_metric": value,
    "epoch": epoch
})

# Log images
wandb.log({"examples": [wandb.Image(img) for img in images]})

# Log histograms
wandb.log({"gradients": wandb.Histogram(grads)})
```

---

## Comparing Experiments

### In Web UI

1. Go to your project page
2. Select multiple runs using checkboxes
3. Click "Compare" button
4. View side-by-side metrics, charts, and configs

### Grouping Experiments

Experiments are automatically grouped by tags. You can also set custom groups:

```bash
python run.py -c ... wandb.group="noise_type_comparison"
```

### Example Grouping Structure

```
rdpp-noising-experiments/
├── Group: noise_types
│   ├── uniform_encoder_greedy
│   ├── gaussian_encoder_greedy
│   └── perlin_encoder_greedy
├── Group: positions
│   ├── uniform_encoder_greedy
│   ├── uniform_projector_greedy
│   └── uniform_mff_oce_greedy
└── Group: sampling
    ├── uniform_encoder_greedy
    ├── uniform_encoder_random
    └── uniform_encoder_kmeans
```

---

## Best Practices

### 1. Use Descriptive Names

```bash
# Good
wandb.name="uniform_encoder_greedy_lr0.005"

# Bad
wandb.name="test1"
```

### 2. Use Tags Extensively

```bash
wandb.tags="['rdpp','noising','uniform','encoder','greedy','mvtec','100epochs']"
```

### 3. Add Notes

```bash
wandb.notes="Testing influence of Perlin noise on encoder features"
```

### 4. Group Related Experiments

```bash
wandb.group="ablation_study_noise_types"
```

### 5. Save Important Artifacts

```python
# Save important files
wandb.save("results/analysis.png")
wandb.save("results/metrics.csv")
```

---

## Troubleshooting

### Error: "wandb: ERROR Error uploading"

**Solution**: Check your internet connection or use offline mode

```bash
export WANDB_MODE=offline
```

### Error: "wandb: ERROR Not logged in"

**Solution**: Login again

```bash
wandb login
```

### Too Many Runs / Storage Limit

**Solution**:
1. Delete old runs from web UI
2. Disable model logging: `wandb.log_model=False`
3. Reduce log frequency: `wandb.log_freq=100`

### Slow Syncing

**Solution**: Use offline mode and sync later

```bash
export WANDB_MODE=offline
# ... run experiments ...
wandb sync --sync-all
```

---

## Advanced Features

### 1. Hyperparameter Sweeps

Create `sweep_config.yaml`:

```yaml
program: run.py
method: grid
parameters:
  model.kwargs.noise_type:
    values: [uniform, gaussian, perlin]
  model.kwargs.noise_position:
    values: [encoder, projector, mff_oce]
  trainer.sampling_method:
    values: [greedy, random, kmeans]
```

Run sweep:

```bash
wandb sweep sweep_config.yaml
wandb agent YOUR_USERNAME/rdpp-noising-experiments/SWEEP_ID
```

### 2. Artifacts (Dataset Versioning)

```python
# Log dataset as artifact
artifact = wandb.Artifact('mvtec-dataset', type='dataset')
artifact.add_dir('data/mvtec')
wandb.log_artifact(artifact)

# Use artifact in another run
artifact = wandb.use_artifact('mvtec-dataset:latest')
artifact_dir = artifact.download()
```

### 3. Reports

Create shareable reports combining runs, charts, and notes:

1. Go to your project
2. Click "Create Report"
3. Add runs, metrics, and markdown notes
4. Share with team or make public

---

## Integration with Existing Scripts

All experiment scripts have been updated to work with wandb:

| Script | wandb Support |
|--------|---------------|
| `run_rdpp_single.sh` | ✅ Auto-names runs |
| `run_rdpp_by_group.sh` | ✅ Auto-tags by group |
| `run_all_rdpp_experiments.sh` | ✅ All 28 runs tracked |

Example wandb dashboard after running all experiments:

```
Project: rdpp-noising-experiments
├── 28 runs total
├── Grouped by: noise_type, position, sampling
├── Metrics: All logged automatically
└── Models: Best checkpoints saved
```

---

## Summary

### Quick Commands

```bash
# Setup
./wandb_setup.sh

# Run with wandb (default)
./run_rdpp_single.sh uniform encoder greedy

# Run without wandb
python run.py -c ... wandb.enable=False

# View experiments
# Visit: https://wandb.ai/YOUR_USERNAME/rdpp-noising-experiments

# Offline mode
export WANDB_MODE=offline
```

### Key Benefits

- All 28 experiments automatically tracked
- No need to manually save logs
- Easy comparison and visualization
- Team collaboration ready
- Model checkpoints versioned
- Free for academic use

---

For more information, visit: [https://docs.wandb.ai/](https://docs.wandb.ai/)
