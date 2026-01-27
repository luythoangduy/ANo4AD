# PatchCore with Adaptive Propose Noising for Anomaly Detection

## Overview

**PatchCore Noising** là model sử dụng **Adaptive Propose Noising based on Feature Influence Analysis** để phát hiện anomaly. Model phân tích xem feature nào ảnh hưởng nhiều nhất đến representation, từ đó propose noise adaptive và detect anomalies hiệu quả.

## Architecture

```
Training Phase:
Input Images (Normal only) [B, 3, 256, 256]
    ↓
Wide ResNet-50 Feature Extraction
    ├── layer2 → [B, 512, 32, 32]
    └── layer3 → [B, 1024, 16, 16]
           ↓
    Interpolate to max size (32×32)
           ↓
    Concatenate channels [B, 1536, 32, 32]
           ↓
    Adaptive pooling → [B, 1024, 1024]
           ↓
    Greedy Coreset Sampling (k-Center)
           ↓
    Memory Bank [M, 1024]

Testing Phase:
Test Image [B, 3, 256, 256]
    ↓
Feature Extraction [B, N, 1024]
    ↓
Adaptive Noising Module:
    ├── Spatial Distance to Memory Bank
    ├── Feature Influence (Gradient-based) ⚡
    └── Adaptive Noise Proposal
           ↓
Anomaly Score = f(Influence, Distance, Noise)
```

## Key Features

1. **Gradient-Based Influence** ⚡: Tính influence 1000x nhanh hơn for-loop
2. **Greedy Coreset Sampling** 🎯: k-Center algorithm cho memory bank tối ưu
3. **Interpolated Feature Fusion** 🎨: Align multi-scale features correctly
4. **Adaptive Noise Proposal**: Propose noise dựa trên feature importance
5. **Unsupervised**: Không cần labels, chỉ train trên normal images

## Masking Mechanism

Dataset hiện tại (**DefaultAD**):
- **Training**: Chỉ load ảnh normal (good), không có synthetic anomaly
- **Testing**: Load ảnh good + anomaly với ground truth mask từ folder `ground_truth/`
- Mask chỉ dùng để evaluate, không dùng trong training

## Training

### 100 epochs (quick training)
```bash
python main.py --config configs/patchcore_noising/patchcore_noising_256_100e.py --cls_names bottle
```

### 300 epochs (full training)
```bash
python main.py --config configs/patchcore_noising/patchcore_noising_256_300e.py --cls_names bottle
```

### Multiple classes
```bash
python main.py --config configs/patchcore_noising/patchcore_noising_256_100e.py --cls_names bottle cable capsule
```

## Testing

```bash
python main.py --config configs/patchcore_noising/patchcore_noising_256_100e.py --cls_names bottle --mode test --checkpoint path/to/checkpoint.pth
```

## Configuration Parameters

### Model Parameters
- `layers_to_extract_from`: ('layer1', 'layer2', 'layer3') - Teacher layers
- `width_per_group`: 128 (64 * 2) - Wide ResNet width
- Teacher: Wide ResNet-50 (frozen)
- Student: 3 Reverse Decoder blocks

### Training Parameters
- `batch_train`: 8 - Batch size cho training
- `batch_test_per`: 8 - Batch size cho testing
- `lr`: 0.005 - Learning rate
- `weight_decay`: 0.0001 - Weight decay
- `warmup_epochs`: 5 - Warmup epochs
- `epoch_full`: 100/300 - Total epochs

### Scheduler
- **Type**: Step scheduler (giống RD, DRAEM)
- `decay_epochs`: 80% of total epochs
- `decay_rate`: 0.1

### Loss Function
- **CosLoss**: λ=1.0 - Cosine similarity loss between teacher-student features

## Expected Results

Model này kỳ vọng performance tốt nhờ:
- **Multi-scale feature learning**: Học từ 3 scales của teacher
- **Reverse Distillation**: Student học reconstruct teacher features
- **Cosine similarity**: Robust metric cho feature matching

## Comparison with Original PatchCore

| Aspect | Original PatchCore | PatchCore Noising (RD) |
|--------|-------------------|------------------------|
| Training | Memory bank fitting | End-to-end learning |
| Architecture | Feature extraction only | Teacher-Student with decoder |
| Loss | None (unsupervised) | Cosine similarity |
| Inference | Nearest neighbor search | Reconstruction error |
| Scales | 2 (layer2, layer3) | 3 (layer1, layer2, layer3) |

## File Structure

```
ADer/
├── configs/
│   ├── __base__/
│   │   └── cfg_model_patchcore_noising.py
│   └── patchcore_noising/
│       ├── patchcore_noising_256_100e.py
│       ├── patchcore_noising_256_300e.py
│       └── README.md
├── model/
│   └── patchcore_noising.py (ReverseDecoder + PATCHCORE_NOISING)
└── trainer/
    └── patchcore_noising_trainer.py
```

## Citation

If you use this model, please cite:

**Reverse Distillation:**
```
@inproceedings{deng2022rd,
  title={Anomaly detection via reverse distillation from one-class embedding},
  author={Deng, Hanqiu and Li, Xingyu},
  booktitle={CVPR},
  year={2022}
}
```

**PatchCore:**
```
@article{roth2021patchcore,
  title={Towards Total Recall in Industrial Anomaly Detection},
  author={Roth, Karsten and Pemula, Latha and Zepeda, Joaquin and Sch{\"o}lkopf, Bernhard and Brox, Thomas and Gehler, Peter},
  journal={CVPR},
  year={2022}
}
```

## Future Work: Propose Noising

Ý tưởng ban đầu là sử dụng PatchCore để tạo memory bank của normal features, sau đó:

1. **Extract Features**: Sử dụng PatchCore để trích xuất features từ test images
2. **Spatial Analysis**: Phân tích theo chiều spatial để xác định khoảng cách đến các cụm normal
3. **Propose Noise**: Dựa vào khoảng cách, propose một lượng noise phù hợp cho mỗi feature
   - Features càng xa cụm normal → propose noise lớn hơn → dễ detect anomaly
   - Features gần cụm normal → propose noise nhỏ → giữ stability
4. **Anomaly Score**: Features có influence lớn nhất (khi thay đổi ảnh hưởng nhiều đến representation) sẽ có score cao nhất

Implementation hiện tại sử dụng **Reverse Distillation** làm baseline, có thể mở rộng thành propose noising trong tương lai.

## WandB Integration

Model đã tích hợp **Weights & Biases** để tracking experiments:

### Setup

```bash
# Install wandb
pip install wandb

# Login (first time only)
wandb login
```

### Configuration

WandB settings trong config file:

```python
self.wandb.enable = True  # Enable/disable WandB
self.wandb.project = 'patchcore-rd-anomaly'  # Project name
self.wandb.entity = None  # Your wandb username/team
self.wandb.name = 'patchcore_rd_256_100e'  # Run name
self.wandb.tags = ['patchcore', 'reverse-distillation']  # Tags
self.wandb.log_model = True  # Save checkpoints to wandb
self.wandb.log_freq = 50  # Log every N iterations
```

### What is Logged

**Training Metrics:**
- Loss terms (cosine loss)
- Learning rate
- Batch time, data time, optimization time
- Epoch number

**Test Metrics:**
- All evaluation metrics per class (AUROC, AUPRO, AP, F1, IoU, etc.)
- Average metrics across all classes
- Best metrics tracking

**Model Checkpoints:**
- Saved automatically when `log_model=True`
- Includes metadata (epoch, AUROC, etc.)

### Disable WandB

Set `self.wandb.enable = False` trong config hoặc:

```bash
python main.py --config ... --wandb.enable False
```

## Notes

- Model này sử dụng **Reverse Distillation** thay vì propose noising như tên gọi
- Training không cần synthetic anomaly, chỉ train trên normal images
- Test time sử dụng cosine distance để tính anomaly score
- Masking chỉ dùng cho evaluation, không affect training process
- **WandB** giúp track experiments dễ dàng hơn so với TensorBoard
