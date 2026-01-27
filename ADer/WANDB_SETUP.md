# WandB Integration for PatchCore Noising

## 🎯 Tổng quan

Đã tích hợp **Weights & Biases (WandB)** vào model PatchCore Noising để tracking experiments một cách dễ dàng và chuyên nghiệp.

## 📦 Cài đặt

```bash
# Install wandb
pip install wandb

# Or from requirements
pip install -r ADer/configs/patchcore_noising/requirements_wandb.txt

# Login (chỉ cần làm 1 lần)
wandb login
```

Khi chạy `wandb login`, bạn sẽ được yêu cầu nhập API key. Lấy API key tại: https://wandb.ai/authorize

## ⚙️ Cấu hình

### 1. Base Config

File [ADer/configs/__base__/cfg_common.py](ADer/configs/__base__/cfg_common.py) đã có WandB config:

```python
# ==> wandb
self.wandb = Namespace()
self.wandb.enable = False  # Enable/disable WandB
self.wandb.project = 'anomaly-detection'  # Project name
self.wandb.entity = None  # Your wandb username/team
self.wandb.name = None  # Run name (auto-generated if None)
self.wandb.tags = []  # Tags for organizing runs
self.wandb.notes = ''  # Notes about the run
self.wandb.log_model = False  # Log model checkpoints to wandb
self.wandb.log_freq = 50  # Log frequency (iterations)
```

### 2. Model-specific Config

File [ADer/configs/patchcore_noising/patchcore_noising_256_100e.py](ADer/configs/patchcore_noising/patchcore_noising_256_100e.py):

```python
# ==> wandb
self.wandb.enable = True  # ✅ Enabled by default
self.wandb.project = 'patchcore-rd-anomaly'
self.wandb.name = f'patchcore_rd_{self.size}_{self.epoch_full}e'
self.wandb.tags = ['patchcore', 'reverse-distillation', 'anomaly-detection']
self.wandb.notes = 'PatchCore with Reverse Distillation for Anomaly Detection'
self.wandb.log_model = True
self.wandb.log_freq = 50
```

## 🚀 Sử dụng

### Quick Start (Linux/Mac)

```bash
cd ADer/configs/patchcore_noising
chmod +x train_with_wandb.sh
./train_with_wandb.sh
```

### Quick Start (Windows)

```cmd
cd ADer\configs\patchcore_noising
train_with_wandb.bat
```

### Manual Training

```bash
# Training với WandB enabled
python main.py \
    --config configs.patchcore_noising.patchcore_noising_256_100e \
    --cls_names bottle \
    --data.root data/mvtec \
    --wandb.enable True \
    --wandb.project my-project \
    --wandb.name my-run-name

# Training với WandB disabled
python main.py \
    --config configs.patchcore_noising.patchcore_noising_256_100e \
    --cls_names bottle \
    --wandb.enable False
```

### Multiple Classes

```bash
python main.py \
    --config configs.patchcore_noising.patchcore_noising_256_100e \
    --cls_names bottle cable capsule \
    --wandb.enable True \
    --wandb.tags patchcore rd multi-class
```

## 📊 Metrics được log

### Training Metrics (mỗi iteration)

- `train/cos`: Cosine similarity loss
- `train/lr`: Learning rate
- `train/batch_t`: Batch processing time
- `train/data_t`: Data loading time
- `train/optim_t`: Optimization time
- `train/epoch`: Current epoch

### Test Metrics (mỗi test epoch)

Per class:
- `test/mAUROC_sp_max_{cls_name}`
- `test/AUROC_sp_{cls_name}`
- `test/mAUROC_px_{cls_name}`
- `test/mAUPRO_px_{cls_name}`
- `test/mAP_sp_max_{cls_name}`
- `test/mAP_px_{cls_name}`
- `test/mF1_max_sp_max_{cls_name}`
- `test/mF1_px_*_{cls_name}`
- `test/mIoU_px_*_{cls_name}`
- ... và nhiều metrics khác

Average (nếu train nhiều classes):
- `test/mAUROC_sp_max_Avg`
- `test/mAUROC_px_Avg`
- ... tất cả metrics average

### Model Checkpoints

Khi `wandb.log_model = True`:
- Checkpoint được save mỗi epoch
- Metadata: epoch, mAUROC_sp_max
- Artifact type: `model`

## 🎨 Customization

### Thay đổi project name

```python
# In config file
self.wandb.project = 'my-awesome-project'

# Or via command line
python main.py --config ... --wandb.project my-awesome-project
```

### Thay đổi run name

```python
# In config file
self.wandb.name = 'experiment-v2-bottle'

# Or via command line
python main.py --config ... --wandb.name experiment-v2-bottle
```

### Thêm tags

```python
# In config file
self.wandb.tags = ['patchcore', 'rd', 'mvtec', 'bottle']

# Or via command line
python main.py --config ... --wandb.tags patchcore rd mvtec
```

### Set entity (team/username)

```python
# In config file
self.wandb.entity = 'my-team'

# Or via command line
python main.py --config ... --wandb.entity my-team
```

## 📁 File Structure

```
ADer/
├── configs/
│   ├── __base__/
│   │   └── cfg_common.py (WandB base config)
│   └── patchcore_noising/
│       ├── patchcore_noising_256_100e.py (WandB enabled)
│       ├── patchcore_noising_256_300e.py (WandB enabled)
│       ├── train_with_wandb.sh (Training script - Linux/Mac)
│       ├── train_with_wandb.bat (Training script - Windows)
│       ├── requirements_wandb.txt (WandB dependencies)
│       └── README.md (Full documentation)
├── trainer/
│   └── patchcore_noising_trainer.py (WandB logging integrated)
└── util/
    └── util.py (WandB utility functions)
```

## 🔧 WandB Utilities

File `util/util.py` cung cấp các functions:

```python
# Initialize WandB
init_wandb(cfg)

# Log metrics
log_wandb(cfg, metrics, step, prefix='train')

# Log images
log_wandb_images(cfg, images_dict, step, prefix='test')

# Save model to WandB
save_wandb_model(cfg, path, metadata=None)

# Finish WandB run
finish_wandb(cfg)
```

## 💡 Tips

1. **Tắt WandB tạm thời**: Set `--wandb.enable False` khi chạy
2. **Offline mode**: Set `WANDB_MODE=offline` trước khi chạy
3. **Resume training**: WandB tự động resume nếu resume_dir được set
4. **Compare runs**: Sử dụng WandB dashboard để so sánh các runs
5. **Sweep**: Có thể tích hợp WandB Sweeps cho hyperparameter tuning

## 🐛 Troubleshooting

### WandB not available

```bash
pip install wandb
wandb login
```

### API key issues

```bash
# Re-login
wandb login --relogin

# Or set API key manually
export WANDB_API_KEY=your_api_key
```

### Disable WandB completely

```bash
# Method 1: Config
python main.py --config ... --wandb.enable False

# Method 2: Environment variable
export WANDB_MODE=disabled
python main.py --config ...
```

## 📚 Tài liệu

- WandB Documentation: https://docs.wandb.ai/
- WandB Python Library: https://docs.wandb.ai/ref/python
- Model README: [ADer/configs/patchcore_noising/README.md](ADer/configs/patchcore_noising/README.md)

## ✨ Features

✅ Automatic logging of all training metrics
✅ Automatic logging of all test metrics
✅ Model checkpoint tracking with metadata
✅ Run resume support
✅ Multi-class support
✅ Easy enable/disable via config
✅ Custom tags and notes
✅ Offline mode support
✅ Team/entity support

Enjoy tracking your experiments! 🎉
