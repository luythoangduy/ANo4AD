#!/bin/bash

# ============================================================================
# Setup Script for RDPP Noising Experiments - Real-IAD Dataset (realiad_1024)
# ============================================================================
# This script will:
# 1. Verify directory structure
# 2. Install required packages
# 3. Prepare Real-IAD realiad_1024 dataset (user-provided or from raw via resize)
# 4. Verify RealIAD structure (per-class JSON + images)
# 5. Install additional dependencies
# 6. Setup Weights & Biases (optional)
# ============================================================================
# Real-IAD: https://realiad4ad.github.io/Real-IAD/
# Dataset access: request realiad_1024 (~53GB) from realiad4ad@outlook.com
# ============================================================================

set -e  # Exit on error

REALIAD_ROOT="data/realiad_1024"
TARGET_RESO=1024

echo "============================================================================"
echo "RDPP Noising Experiment Setup - Real-IAD (realiad_1024)"
echo "Started at: $(date)"
echo "============================================================================"
echo ""

# ============================================================================
# Step 1: Verify Directory Structure
# ============================================================================
if [ ! -f "run.py" ]; then
    echo "Error: run.py not found. Make sure you're in the ADer directory."
    exit 1
fi

echo "[Step 1/7] Verifying directory structure..."
echo "Current directory: $(pwd)"
echo ""

# ============================================================================
# Step 2: Install Core Python Packages
# ============================================================================
echo "[Step 2/7] Installing core Python packages..."
pip install --upgrade pip

# pandas required for data processing
pip install pandas

# Install perlin noise library for experiments
pip install perlin-numpy

echo "Core packages installed."
echo ""

# ============================================================================
# Step 3: Prepare realiad_1024 Dataset
# ============================================================================
echo "[Step 3/7] Preparing Real-IAD realiad_1024 dataset..."
echo ""
echo "Real-IAD realiad_1024 is 1024x1024 resolution (~53GB)."
echo "Obtain it by requesting access from: realiad4ad@outlook.com"
echo "See: https://realiad4ad.github.io/Real-IAD/ or https://github.com/TencentYoutuResearch/AnomalyDetection_Real-IAD"
echo ""

if [ -d "$REALIAD_ROOT" ] && [ "$(ls -A $REALIAD_ROOT 2>/dev/null)" ]; then
    echo "Found existing $REALIAD_ROOT - skipping download/extract."
else
    echo "Choose how to set up realiad_1024:"
    echo "  1) I have the realiad_1024 folder (or archive) - I will provide the path"
    echo "  2) I have raw Real-IAD (explicit_full) - create realiad_1024 via resize_realiad.py"
    echo "  3) Skip - I will manually place data in $REALIAD_ROOT later"
    read -p "Enter choice (1/2/3): " choice

    case "$choice" in
        1)
            read -p "Path to realiad_1024 folder or archive (.tar / .tar.gz / .zip): " user_path
            if [ -z "$user_path" ] || [ ! -e "$user_path" ]; then
                echo "Invalid or empty path. Skipping. Please extract/copy realiad_1024 to $REALIAD_ROOT manually."
            else
                mkdir -p data
                if [ -d "$user_path" ]; then
                    echo "Copying folder to $REALIAD_ROOT..."
                    cp -r "$user_path" "$REALIAD_ROOT"
                elif [ -f "$user_path" ]; then
                    echo "Extracting archive to data/..."
                    mkdir -p temp_download
                    case "$user_path" in
                        *.zip)   unzip -q -o "$user_path" -d temp_download ;;
                        *.tar)   tar -xf "$user_path" -C temp_download ;;
                        *.tar.gz|*.tgz) tar -xzf "$user_path" -C temp_download ;;
                        *)       echo "Unsupported archive format. Copy or extract manually to $REALIAD_ROOT"; rm -rf temp_download 2>/dev/null; exit 1 ;;
                    esac
                    # Move extracted content to realiad_1024 (may be realiad_1024 subdir or direct content)
                    if [ -d "temp_download/realiad_1024" ]; then
                        mv temp_download/realiad_1024 "$REALIAD_ROOT"
                    else
                        mkdir -p "$REALIAD_ROOT"
                        mv temp_download/* "$REALIAD_ROOT/" 2>/dev/null || true
                    fi
                    rm -rf temp_download
                fi
                echo "Done."
            fi
            ;;
        2)
            read -p "Path to raw Real-IAD root containing explicit_full (e.g. data/realiad or /path/to/realiad): " raw_root
            if [ -z "$raw_root" ] || [ ! -d "$raw_root/explicit_full" ]; then
                echo "Path must contain explicit_full/ subdir. Skipping. You can run later:"
                echo "  python data/gen_benchmark/resize_realiad.py --root_path <path>/explicit_full --target_root data/ --target_reso $TARGET_RESO"
            else
                echo "Running resize_realiad.py (target_reso=$TARGET_RESO)..."
                pip install Pillow -q
                python data/gen_benchmark/resize_realiad.py \
                    --root_path "$raw_root/explicit_full" \
                    --target_root data/ \
                    --target_reso $TARGET_RESO
                # resize_realiad.py writes to data/realiad_1024/
                echo "Done. Data written to $REALIAD_ROOT"
            fi
            ;;
        3)
            echo "Skipping. Create $REALIAD_ROOT and place realiad_1024 content there when ready."
            mkdir -p "$REALIAD_ROOT"
            ;;
        *)
            echo "Invalid choice. Create $REALIAD_ROOT and add data manually."
            mkdir -p "$REALIAD_ROOT"
            ;;
    esac
fi

echo ""

# ============================================================================
# Step 4: Verify RealIAD Structure
# ============================================================================
echo "[Step 4/7] Verifying RealIAD structure..."
# RealIAD uses per-class JSON files (e.g. audiojack.json) and image dirs under each class
json_count=$(find "$REALIAD_ROOT" -maxdepth 1 -name "*.json" 2>/dev/null | wc -l)
if [ "$json_count" -gt 0 ]; then
    echo "Found $json_count class JSON file(s) in $REALIAD_ROOT."
else
    echo "Note: No .json files found in $REALIAD_ROOT. Ensure you have Real-IAD format (per-class .json + images)."
fi
echo ""

# ============================================================================
# Step 5: Install Additional Dependencies
# ============================================================================
echo "[Step 5/7] Installing additional dependencies..."

# Install PyTorch dependencies (adjust based on your CUDA version)
# Uncomment the appropriate line for your system:

# For CUDA 11.8
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# For CPU only
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install other required packages
pip install adeval \
    FrEIA \
    geomloss \
    ninja \
    faiss-cpu \
    einops \
    numba \
    imgaug \
    scikit-image \
    opencv-python \
    fvcore \
    tensorboardX \
    timm
pip install numpy==1.26.4 scikit-learn wandb

conda install -c conda-forge accimage -y
echo "Additional dependencies installed."
echo ""

# ============================================================================
# Step 6: Setup Weights & Biases (Optional)
# ============================================================================
echo "[Step 6/7] Setting up Weights & Biases (optional)..."
echo ""
echo "Weights & Biases (wandb) is used for experiment tracking and visualization."
echo "wandb is enabled by default in the config."
echo ""
read -p "Do you want to setup wandb now? (y/n): " setup_wandb

if [ "$setup_wandb" = "y" ] || [ "$setup_wandb" = "Y" ]; then
    echo ""
    echo "Starting wandb setup..."
    chmod +x wandb_setup.sh 2>/dev/null || true
    ./wandb_setup.sh 2>/dev/null || echo "Run ./wandb_setup.sh manually if needed."
else
    echo ""
    echo "Skipping wandb setup."
    echo "You can run './wandb_setup.sh' later to configure wandb."
    echo ""
    echo "Note: wandb is enabled by default. To disable it for a run:"
    echo "  python run.py -c configs/rdpp_noising/rdpp_noising_256_100e_realiad.py wandb.enable=False"
fi

echo ""

# ============================================================================
# Step 7: Verify Installation
# ============================================================================
echo "[Step 7/7] Verifying installation..."
echo "============================================================================"
echo "Verifying installation..."
echo "============================================================================"
echo ""

# Check if data exists
if [ -d "$REALIAD_ROOT" ] && [ "$(ls -A $REALIAD_ROOT 2>/dev/null)" ]; then
    echo "✓ Real-IAD realiad_1024: OK ($REALIAD_ROOT)"
else
    echo "✗ Real-IAD realiad_1024: NOT FOUND or empty (expected: $REALIAD_ROOT)"
fi

# Check for class JSON files
json_count=$(find "$REALIAD_ROOT" -maxdepth 1 -name "*.json" 2>/dev/null | wc -l)
if [ "$json_count" -gt 0 ]; then
    echo "✓ RealIAD class JSON files: OK ($json_count classes)"
else
    echo "✗ RealIAD class JSON files: NONE (add per-class .json from Real-IAD)"
fi

# Check if model directory exists
if [ -d "model" ]; then
    echo "✓ Model directory: OK"
else
    echo "✗ Model directory: NOT FOUND"
fi

# Test Python imports
echo ""
echo "Testing Python package imports..."
python -c "import torch; print('✓ PyTorch:', torch.__version__)" || echo "✗ PyTorch: FAILED"
python -c "import timm; print('✓ timm:', timm.__version__)" || echo "✗ timm: FAILED"
python -c "import pandas; print('✓ pandas: OK')" || echo "✗ pandas: FAILED"
python -c "from perlin_numpy import generate_perlin_noise_2d; print('✓ perlin-numpy: OK')" || echo "✗ perlin-numpy: FAILED"
python -c "import geomloss; print('✓ geomloss: OK')" || echo "✗ geomloss: FAILED"
python -c "import faiss; print('✓ faiss: OK')" || echo "✗ faiss: FAILED"

echo ""
echo "============================================================================"
echo "Setup Complete (Real-IAD realiad_1024)!"
echo "============================================================================"
echo ""
echo "Next steps:"
echo "1. Download pretrained weights (if needed):"
echo "   - Wide ResNet50: model/pretrain/wide_resnet50_racm-8234f177.pth"
echo ""
echo "2. Use a config that points to realiad_1024, e.g. create or use:"
echo "   configs/rdpp_noising/rdpp_noising_256_100e_realiad.py"
echo "   with: self.data.type = 'RealIAD', self.data.root = 'data/realiad_1024'"
echo ""
echo "3. Run training on Real-IAD (realiad_1024):"
echo "   python run.py -c configs/rdpp_noising/rdpp_noising_256_100e_realiad.py"
echo ""
echo "4. Request Real-IAD access if you have not: realiad4ad@outlook.com"
echo "   Project: https://realiad4ad.github.io/Real-IAD/"
echo ""
echo "For more information, see: data/README.md (Real-IAD section)"
echo ""
echo "Finished at: $(date)"
echo "============================================================================"
