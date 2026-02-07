#!/bin/bash

# ============================================================================
# RDPP Noising Single Experiment Runner
# ============================================================================
# Usage: ./run_rdpp_single.sh <noise_type> <noise_position> <sampling_method>
#
# Examples:
#   ./run_rdpp_single.sh uniform encoder greedy
#   ./run_rdpp_single.sh gaussian projector random
#   ./run_rdpp_single.sh perlin mff_oce kmeans
#   ./run_rdpp_single.sh none none none  # No noise baseline
# ============================================================================

set -e

# Configuration
CONFIG_FILE="configs/rdpp_noising/rdpp_noising_256_100e.py"
GPU_ID=0

# Parse arguments
NOISE_TYPE=${1:-"uniform"}
NOISE_POSITION=${2:-"encoder"}
SAMPLING_METHOD=${3:-"greedy"}

# Check if no noise mode
if [ "$NOISE_TYPE" = "none" ]; then
    ENABLE_NOISE="False"
    EXP_NAME="no_noise_baseline"
    echo "============================================================================"
    echo "Running RDPP Noising Experiment: No Noise (Baseline)"
    echo "============================================================================"
else
    ENABLE_NOISE="True"
    EXP_NAME="${NOISE_TYPE}_${NOISE_POSITION}_${SAMPLING_METHOD}"
    echo "============================================================================"
    echo "Running RDPP Noising Experiment"
    echo "============================================================================"
    echo "Noise Type: $NOISE_TYPE"
    echo "Noise Position: $NOISE_POSITION"
    echo "Sampling Method: $SAMPLING_METHOD"
fi

echo "Experiment Name: $EXP_NAME"
echo "Started at: $(date)"
echo "============================================================================"
echo ""

# Run experiment
if [ "$ENABLE_NOISE" = "False" ]; then
    CUDA_VISIBLE_DEVICES=$GPU_ID python run.py \
        -c "$CONFIG_FILE" \
        -m train \
        model.kwargs.enable_noise=False \
        trainer.logdir_sub="$EXP_NAME" \
        wandb.name="$EXP_NAME" \
        wandb.tags="['rdpp','noising','no-noise','baseline']"
else
    CUDA_VISIBLE_DEVICES=$GPU_ID python run.py \
        -c "$CONFIG_FILE" \
        -m train \
        model.kwargs.enable_noise=True \
        model.kwargs.noise_type="$NOISE_TYPE" \
        model.kwargs.noise_position="$NOISE_POSITION" \
        trainer.sampling_method="$SAMPLING_METHOD" \
        trainer.logdir_sub="$EXP_NAME" \
        wandb.name="$EXP_NAME" \
        wandb.tags="['rdpp','noising','$NOISE_TYPE','$NOISE_POSITION','$SAMPLING_METHOD']"
fi

echo ""
echo "============================================================================"
echo "Experiment Completed!"
echo "Finished at: $(date)"
echo "============================================================================"
