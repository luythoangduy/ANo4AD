# RD++ Noising Configuration

This folder contains configuration files for RD++ (Revisit RD) with Adaptive Noising.

## Model Architecture

RD++ Noising combines:
- **RD++ features**: Multi-projection layer, SSOT loss, reconstruct loss, contrast loss
- **Adaptive Noising**: Memory bank based influence analysis for noise injection

## Two-Phase Training

1. **Phase 1 - Memory Bank Construction**: Extract teacher features and build memory bank
2. **Phase 2 - Training**: Train decoder with adaptive noise + RD++ projection losses

## Configuration Files

- `rdpp_noising_256_100e.py`: 256x256 resolution, 100 epochs
- `rdpp_noising_256_300e.py`: 256x256 resolution, 300 epochs

## Key Parameters

### Memory Bank
- `coreset_sampling_ratio`: Ratio of features to keep (default: 0.01 = 1%)
- `sampling_method`: 'auto', 'greedy', 'random', or 'hybrid'
- `coreset_device`: 'auto', 'cpu', or 'cuda'

### Noise Configuration
- `n_neighbors`: K nearest neighbors for influence computation
- `noise_std_range`: Range of noise standard deviation (min, max)
- `enable_noise`: Enable/disable adaptive noise

### RD++ Specific
- `proj_base`: Base channel for projection layer (default: 64)

## Usage

```bash
python run.py --cfg configs/rdpp_noising/rdpp_noising_256_100e.py
```
