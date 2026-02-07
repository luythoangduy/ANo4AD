# ============================================================================
# Weights & Biases (wandb) Setup Script (Windows PowerShell)
# ============================================================================

Write-Host "============================================================================" -ForegroundColor Cyan
Write-Host "Weights & Biases (wandb) Setup" -ForegroundColor Cyan
Write-Host "============================================================================" -ForegroundColor Cyan
Write-Host ""

# Step 1: Install wandb
Write-Host "[Step 1/3] Installing wandb..." -ForegroundColor Yellow
pip install wandb

Write-Host "wandb installed successfully!" -ForegroundColor Green
Write-Host ""

# Step 2: Login to wandb
Write-Host "[Step 2/3] Logging in to Weights & Biases..." -ForegroundColor Yellow
Write-Host ""
Write-Host "You have two options:"
Write-Host "  1. Login with API key (recommended for servers/clusters)"
Write-Host "  2. Login with browser (recommended for local development)"
Write-Host ""
$login_method = Read-Host "Choose login method (1 or 2)"

if ($login_method -eq "1") {
    Write-Host ""
    Write-Host "Please enter your wandb API key."
    Write-Host "You can find your API key at: https://wandb.ai/authorize"
    Write-Host ""
    $api_key = Read-Host "API Key"
    wandb login $api_key
} elseif ($login_method -eq "2") {
    Write-Host ""
    Write-Host "Opening browser for wandb login..." -ForegroundColor Yellow
    wandb login
} else {
    Write-Host "Invalid choice. Please run the script again." -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "Successfully logged in to wandb!" -ForegroundColor Green
Write-Host ""

# Step 3: Configure wandb settings
Write-Host "[Step 3/3] Configuring wandb settings..." -ForegroundColor Yellow
Write-Host ""

# Ask for wandb entity (username/team)
Write-Host "Enter your wandb username or team name (leave empty to use default):"
$entity = Read-Host "Entity"

if ($entity) {
    Write-Host "Setting wandb entity to: $entity" -ForegroundColor Green
    Write-Host ""
    Write-Host "To use this entity, update your config file:"
    Write-Host "  self.wandb.entity = '$entity'"
    Write-Host ""
}

# Ask for project name
$use_default_project = Read-Host "Use default project name 'rdpp-noising-experiments'? (y/n)"

if ($use_default_project -ne "y") {
    $project_name = Read-Host "Enter custom project name"
    Write-Host ""
    Write-Host "To use this project, update your config file:"
    Write-Host "  self.wandb.project = '$project_name'"
    Write-Host ""
}

# Test wandb
Write-Host "Testing wandb connection..." -ForegroundColor Yellow
python -c "import wandb; print('✓ wandb version:', wandb.__version__)"

Write-Host ""
Write-Host "============================================================================" -ForegroundColor Cyan
Write-Host "wandb Setup Complete!" -ForegroundColor Green
Write-Host "============================================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host ""
Write-Host "1. Verify wandb config in: configs\rdpp_noising\rdpp_noising_256_100e.py"
Write-Host "   - Set self.wandb.entity = '$entity' (if you have a team)"
Write-Host "   - wandb.enable is already set to True"
Write-Host ""
Write-Host "2. Run an experiment to test wandb integration:"
Write-Host "   run_rdpp_single.bat uniform encoder greedy"
Write-Host ""
Write-Host "3. View your experiments at: https://wandb.ai/$entity/rdpp-noising-experiments"
Write-Host ""
Write-Host "4. To disable wandb for a specific run, add:"
Write-Host "   wandb.enable=False"
Write-Host "   Example: python run.py -c ... wandb.enable=False"
Write-Host ""
Write-Host "5. Optional: Set wandb mode"
Write-Host "   - Online mode (default): experiments sync to cloud"
Write-Host "   - Offline mode: `$env:WANDB_MODE='offline'"
Write-Host "   - Disabled mode: `$env:WANDB_MODE='disabled'"
Write-Host ""
Write-Host "============================================================================" -ForegroundColor Cyan
