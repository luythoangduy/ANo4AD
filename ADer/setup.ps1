# ============================================================================
# Setup Script for RDPP Noising Experiments (Windows PowerShell)
# ============================================================================

Write-Host "============================================================================" -ForegroundColor Cyan
Write-Host "RDPP Noising Experiment Setup (Windows)" -ForegroundColor Cyan
Write-Host "Started at: $(Get-Date)" -ForegroundColor Cyan
Write-Host "============================================================================" -ForegroundColor Cyan
Write-Host ""

# ============================================================================
# Step 1: Verify Directory
# ============================================================================
Write-Host "[Step 1/6] Verifying directory structure..." -ForegroundColor Yellow

if (-not (Test-Path "run.py")) {
    Write-Host "Error: run.py not found. Make sure you're in the ADer directory." -ForegroundColor Red
    exit 1
}

Write-Host "Current directory: $(Get-Location)" -ForegroundColor Green
Write-Host ""

# ============================================================================
# Step 2: Install Core Python Packages
# ============================================================================
Write-Host "[Step 2/6] Installing core Python packages..." -ForegroundColor Yellow

# Upgrade pip
python -m pip install --upgrade pip

# Install gdown
pip install gdown

# Install perlin noise
pip install perlin-numpy

Write-Host "Core packages installed." -ForegroundColor Green
Write-Host ""

# ============================================================================
# Step 3: Download MVTec AD Dataset
# ============================================================================
Write-Host "[Step 3/6] Downloading MVTec AD dataset..." -ForegroundColor Yellow
Write-Host "This may take several minutes depending on your internet connection..." -ForegroundColor Yellow

# Create temporary download directory
New-Item -ItemType Directory -Force -Path "temp_download" | Out-Null
Set-Location "temp_download"

# Download dataset
gdown --fuzzy "https://drive.google.com/file/d/1JhhA36qmH8lKCgiX9lU6v8D7B1Y3Xa7r/view?usp=drive_link" -O datasets.zip

Write-Host "Download completed." -ForegroundColor Green
Write-Host ""

# ============================================================================
# Step 4: Extract and Organize Dataset
# ============================================================================
Write-Host "[Step 4/6] Extracting dataset..." -ForegroundColor Yellow

Expand-Archive -Path "datasets.zip" -DestinationPath "." -Force

Write-Host "Creating data directory structure..." -ForegroundColor Yellow
Set-Location ".."
New-Item -ItemType Directory -Force -Path "data\mvtec" | Out-Null

Write-Host "Moving dataset files..." -ForegroundColor Yellow
Move-Item -Path "temp_download\datasets\mvtec_anomaly_detection\*" -Destination "data\mvtec\" -Force

Write-Host "Cleaning up temporary files..." -ForegroundColor Yellow
Remove-Item -Path "temp_download" -Recurse -Force

Write-Host "Dataset setup completed." -ForegroundColor Green
Write-Host ""

# ============================================================================
# Step 5: Generate Benchmark Metadata
# ============================================================================
Write-Host "[Step 5/6] Generating benchmark metadata..." -ForegroundColor Yellow
python data\gen_benchmark\mvtec.py

Write-Host "Benchmark metadata generated." -ForegroundColor Green
Write-Host ""

# ============================================================================
# Step 6: Install Additional Dependencies
# ============================================================================
Write-Host "[Step 6/6] Installing additional dependencies..." -ForegroundColor Yellow

# Install PyTorch (adjust based on your CUDA version)
# For CUDA 11.8
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# For CPU only
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install other packages
pip install adeval FrEIA geomloss ninja faiss-cpu einops numba imgaug scikit-image opencv-python fvcore tensorboardX timm

Write-Host "Additional dependencies installed." -ForegroundColor Green
Write-Host ""

# ============================================================================
# Verify Installation
# ============================================================================
Write-Host "============================================================================" -ForegroundColor Cyan
Write-Host "Verifying installation..." -ForegroundColor Cyan
Write-Host "============================================================================" -ForegroundColor Cyan
Write-Host ""

# Check if data exists
if (Test-Path "data\mvtec" -and (Get-ChildItem "data\mvtec").Count -gt 0) {
    Write-Host "✓ MVTec dataset: OK" -ForegroundColor Green
} else {
    Write-Host "✗ MVTec dataset: NOT FOUND" -ForegroundColor Red
}

# Check if meta.json exists
if (Test-Path "data\mvtec\meta.json") {
    Write-Host "✓ Benchmark metadata: OK" -ForegroundColor Green
} else {
    Write-Host "✗ Benchmark metadata: NOT FOUND" -ForegroundColor Red
}

# Check model directory
if (Test-Path "model") {
    Write-Host "✓ Model directory: OK" -ForegroundColor Green
} else {
    Write-Host "✗ Model directory: NOT FOUND" -ForegroundColor Red
}

# Test Python imports
Write-Host ""
Write-Host "Testing Python package imports..." -ForegroundColor Yellow
python -c "import torch; print('✓ PyTorch:', torch.__version__)"
python -c "import timm; print('✓ timm:', timm.__version__)"
python -c "from perlin_numpy import generate_perlin_noise_2d; print('✓ perlin-numpy: OK')"
python -c "import geomloss; print('✓ geomloss: OK')"
python -c "import faiss; print('✓ faiss: OK')"

Write-Host ""
Write-Host "============================================================================" -ForegroundColor Cyan
Write-Host "Setup Complete!" -ForegroundColor Green
Write-Host "============================================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "1. Download pretrained weights (if needed):"
Write-Host "   - Wide ResNet50: model\pretrain\wide_resnet50_racm-8234f177.pth"
Write-Host ""
Write-Host "2. Test with a quick experiment:"
Write-Host "   python run.py -c configs\rdpp_noising\rdpp_noising_256_100e.py -m train model.kwargs.enable_noise=False"
Write-Host ""
Write-Host "3. Run experiments using Python directly (Windows):"
Write-Host "   python run.py -c configs\rdpp_noising\rdpp_noising_256_100e.py -m train model.kwargs.enable_noise=True model.kwargs.noise_type=`"uniform`" model.kwargs.noise_position=`"encoder`" trainer.sampling_method=`"greedy`""
Write-Host ""
Write-Host "For more information, see: RDPP_EXPERIMENTS_README.md"
Write-Host ""
Write-Host "Finished at: $(Get-Date)" -ForegroundColor Cyan
Write-Host "============================================================================" -ForegroundColor Cyan
