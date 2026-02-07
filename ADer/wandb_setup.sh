#!/bin/bash

# ============================================================================
# Weights & Biases (wandb) Setup Script
# ============================================================================
# This script helps you set up wandb for experiment tracking
# ============================================================================

set -e

echo "============================================================================"
echo "Weights & Biases (wandb) Setup"
echo "============================================================================"
echo ""

# Step 1: Install wandb
echo "[Step 1/3] Installing wandb..."
pip install wandb

echo "wandb installed successfully!"
echo ""

# Step 2: Login to wandb
echo "[Step 2/3] Logging in to Weights & Biases..."
echo ""
echo "You have two options:"
echo "  1. Login with API key (recommended for servers/clusters)"
echo "  2. Login with browser (recommended for local development)"
echo ""
read -p "Choose login method (1 or 2): " login_method

if [ "$login_method" = "1" ]; then
    echo ""
    echo "Please enter your wandb API key."
    echo "You can find your API key at: https://wandb.ai/authorize"
    echo ""
    read -p "API Key: " api_key
    wandb login "$api_key"
elif [ "$login_method" = "2" ]; then
    echo ""
    echo "Opening browser for wandb login..."
    wandb login
else
    echo "Invalid choice. Please run the script again."
    exit 1
fi

echo ""
echo "Successfully logged in to wandb!"
echo ""

# Step 3: Configure wandb settings
echo "[Step 3/3] Configuring wandb settings..."
echo ""

# Ask for wandb entity (username/team)
echo "Enter your wandb username or team name (leave empty to use default):"
read -p "Entity: " entity

if [ -n "$entity" ]; then
    echo "Setting wandb entity to: $entity"
    echo ""
    echo "To use this entity, update your config file:"
    echo "  self.wandb.entity = '$entity'"
    echo ""
fi

# Ask for project name
read -p "Use default project name 'rdpp-noising-experiments'? (y/n): " use_default_project

if [ "$use_default_project" != "y" ]; then
    read -p "Enter custom project name: " project_name
    echo ""
    echo "To use this project, update your config file:"
    echo "  self.wandb.project = '$project_name'"
    echo ""
fi

# Test wandb
echo "Testing wandb connection..."
python -c "import wandb; print('✓ wandb version:', wandb.__version__)"

echo ""
echo "============================================================================"
echo "wandb Setup Complete!"
echo "============================================================================"
echo ""
echo "Next steps:"
echo ""
echo "1. Verify wandb config in: configs/rdpp_noising/rdpp_noising_256_100e.py"
echo "   - Set self.wandb.entity = '$entity' (if you have a team)"
echo "   - wandb.enable is already set to True"
echo ""
echo "2. Run an experiment to test wandb integration:"
echo "   ./run_rdpp_single.sh uniform encoder greedy"
echo ""
echo "3. View your experiments at: https://wandb.ai/$entity/rdpp-noising-experiments"
echo ""
echo "4. To disable wandb for a specific run, add:"
echo "   wandb.enable=False"
echo "   Example: python run.py -c ... wandb.enable=False"
echo ""
echo "5. Optional: Set wandb mode"
echo "   - Online mode (default): experiments sync to cloud"
echo "   - Offline mode: export WANDB_MODE=offline"
echo "   - Disabled mode: export WANDB_MODE=disabled"
echo ""
echo "============================================================================"
