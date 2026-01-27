# RD with Adaptive Noising (RD-Noising)

## Overview

RD-Noising combines the Reverse Distillation (RD) framework with adaptive noise injection based on memory bank influence analysis. This creates a two-phase training approach:

### Phase 1: Memory Bank Construction
- Extract teacher features from all training (normal) images
- Apply greedy k-Center coreset sampling to build compact memory banks
- Memory banks are built for each feature scale (layer1, layer2, layer3)

### Phase 2: Training with Adaptive Noise
- Compute influence scores based on distance to memory bank
- Generate adaptive noise proportional to feature influence
- Train decoder (student) network with noised teacher features
- Knowledge distillation with cosine similarity loss

## Key Components

### 1. Memory Bank
- Built from teacher network features (frozen)
- Uses greedy coreset sampling (k-Center algorithm) for efficient representation
- Multi-scale: separate memory banks for each feature layer

### 2. Adaptive Noising Module
- Computes feature influence based on distance to nearest neighbors in memory bank
- Generates noise std proportional to:
  - Distance to normal clusters (far = higher noise)
  - Per-dimension influence (sensitive dimensions = higher noise)
- Learnable weights for influence computation

### 3. Two-Phase Training
- Phase 1: One-time memory bank construction before training
- Phase 2: Standard RD training with adaptive noise injection

## Configuration

```python
# Model configuration
self.model.kwargs = dict(
    # Memory bank
    coreset_sampling_ratio=0.01,  # 1% of features kept
    
    # Adaptive noise
    n_neighbors=9,                 # K nearest neighbors
    noise_std_range=(0.01, 0.3),  # Min/max noise std
    noise_layers=[0, 1, 2],       # Which layers to apply noise
    enable_noise=True,            # Enable/disable noise
)

# Trainer configuration
self.trainer.noise_enabled = True
self.trainer.noise_warmup_epochs = 0  # Epochs before enabling noise
```

## Usage

### Training

```bash
# 100 epochs
python run.py -c configs/rd_noising/rd_noising_256_100e.py

# 300 epochs with noise warmup
python run.py -c configs/rd_noising/rd_noising_256_300e.py
```

### Running on specific class
```bash
python run.py -c configs/rd_noising/rd_noising_256_100e.py \
    data.cls_names="['bottle']"
```

## Architecture

```
Input Image
    │
    ▼
┌─────────────────┐
│ Teacher Network │ (Frozen WideResNet50)
│ (Feature Extractor)
└─────────────────┘
    │
    ├── layer1 features → Memory Bank 1 → Adaptive Noise 1
    ├── layer2 features → Memory Bank 2 → Adaptive Noise 2  
    └── layer3 features → Memory Bank 3 → Adaptive Noise 3
    │
    ▼
┌─────────────────┐
│ MFF-OCE Module  │ (Multi-scale Feature Fusion)
└─────────────────┘
    │
    ▼
┌─────────────────┐
│ Student Decoder │ (Trainable)
└─────────────────┘
    │
    ▼
Reconstructed Features → Cosine Loss with Teacher
```

## Differences from Original RD

| Aspect | RD | RD-Noising |
|--------|-----|------------|
| Memory Bank | No | Yes (Phase 1) |
| Noise Injection | No | Adaptive based on influence |
| Training | Single phase | Two phases |
| Feature Augmentation | None | Influence-based noise |

## Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `coreset_sampling_ratio` | 0.01 | Fraction of features to keep in memory bank |
| `n_neighbors` | 9 | Number of nearest neighbors for influence |
| `noise_std_range` | (0.01, 0.3) | Range of noise standard deviation |
| `noise_layers` | [0, 1, 2] | Layer indices to apply noise |
| `noise_warmup_epochs` | 0 | Epochs before enabling noise |

## Files

- `model/rd_noising.py` - Main model implementation
- `trainer/rd_noising_trainer.py` - Two-phase trainer
- `configs/__base__/cfg_model_rd_noising.py` - Base model config
- `configs/rd_noising/*.py` - Experiment configs
