#!/bin/bash

# ============================================================================
# RDPP Noising Full Experiments - 100 Epochs
# ============================================================================
#
# Experiment Dimensions:
# 1. Noise Types: none, uniform, gaussian, perlin
# 2. Noise Positions: encoder, projector, mff_oce
# 3. Sampling Methods: greedy, random, kmeans
#
# Total Experiments:
# - No noise: 1 experiment (no noise, sampling doesn't matter)
# - With noise: 3 noise_types × 3 positions × 3 sampling = 27 experiments
# Total: 28 experiments
# ============================================================================

set -e  # Exit on error

# Configuration
CONFIG_FILE="configs/rdpp_noising/rdpp_noising_256_100e.py"
EPOCHS=100
GPU_ID=0

# Create results directory
RESULTS_DIR="results/rdpp_experiments_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_DIR"

# Log file
LOG_FILE="$RESULTS_DIR/experiments.log"

echo "============================================================================" | tee -a "$LOG_FILE"
echo "RDPP Noising Full Experiments - 100 Epochs" | tee -a "$LOG_FILE"
echo "Started at: $(date)" | tee -a "$LOG_FILE"
echo "============================================================================" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Counter
TOTAL_EXPS=28
CURRENT_EXP=0

# ============================================================================
# Experiment 1: No Noise (Baseline)
# ============================================================================
CURRENT_EXP=$((CURRENT_EXP + 1))
echo "[$CURRENT_EXP/$TOTAL_EXPS] Running: No Noise (Baseline)" | tee -a "$LOG_FILE"
echo "Started at: $(date)" | tee -a "$LOG_FILE"

CUDA_VISIBLE_DEVICES=$GPU_ID python run.py \
    -c "$CONFIG_FILE" \
    -m train \
    model.kwargs.enable_noise=False \
    trainer.noise_enabled=False \
    trainer.logdir_sub="no_noise_baseline" \
    wandb.name="no_noise_baseline" \
    2>&1 | tee "$RESULTS_DIR/exp_${CURRENT_EXP}_no_noise.log"

echo "Completed at: $(date)" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# ============================================================================
# Experiments 2-28: With Noise
# ============================================================================

# Noise types to test
NOISE_TYPES=("uniform" "gaussian" "perlin")

# Noise positions to test
NOISE_POSITIONS=("encoder" "projector" "mff_oce")

# Sampling methods to test
SAMPLING_METHODS=("greedy" "random" "kmeans")

# Loop through all combinations
for noise_type in "${NOISE_TYPES[@]}"; do
    for noise_pos in "${NOISE_POSITIONS[@]}"; do
        for sampling in "${SAMPLING_METHODS[@]}"; do
            CURRENT_EXP=$((CURRENT_EXP + 1))

            EXP_NAME="${noise_type}_${noise_pos}_${sampling}"

            echo "============================================================================" | tee -a "$LOG_FILE"
            echo "[$CURRENT_EXP/$TOTAL_EXPS] Running: $EXP_NAME" | tee -a "$LOG_FILE"
            echo "  Noise Type: $noise_type" | tee -a "$LOG_FILE"
            echo "  Noise Position: $noise_pos" | tee -a "$LOG_FILE"
            echo "  Sampling Method: $sampling" | tee -a "$LOG_FILE"
            echo "  Started at: $(date)" | tee -a "$LOG_FILE"

            CUDA_VISIBLE_DEVICES=$GPU_ID python run.py \
                -c "$CONFIG_FILE" \
                -m train \
                model.kwargs.enable_noise=True \
                model.kwargs.noise_type="$noise_type" \
                model.kwargs.noise_position="$noise_pos" \
                trainer.sampling_method="$sampling" \
                trainer.logdir_sub="$EXP_NAME" \
                wandb.name="$EXP_NAME" \
                2>&1 | tee "$RESULTS_DIR/exp_${CURRENT_EXP}_${EXP_NAME}.log"

            echo "  Completed at: $(date)" | tee -a "$LOG_FILE"
            echo "" | tee -a "$LOG_FILE"
        done
    done
done

# ============================================================================
# Summary
# ============================================================================
echo "============================================================================" | tee -a "$LOG_FILE"
echo "All Experiments Completed!" | tee -a "$LOG_FILE"
echo "Total experiments run: $TOTAL_EXPS" | tee -a "$LOG_FILE"
echo "Finished at: $(date)" | tee -a "$LOG_FILE"
echo "Results saved in: $RESULTS_DIR" | tee -a "$LOG_FILE"
echo "============================================================================" | tee -a "$LOG_FILE"

# Generate summary report
echo "" | tee -a "$LOG_FILE"
echo "Experiment Summary:" | tee -a "$LOG_FILE"
echo "==================" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "1. No Noise Baseline: 1 experiment" | tee -a "$LOG_FILE"
echo "2. Noise Type Variations:" | tee -a "$LOG_FILE"
echo "   - Uniform Noise: 9 experiments (3 positions × 3 sampling)" | tee -a "$LOG_FILE"
echo "   - Gaussian Noise: 9 experiments (3 positions × 3 sampling)" | tee -a "$LOG_FILE"
echo "   - Perlin Noise: 9 experiments (3 positions × 3 sampling)" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "3. Noise Position Variations:" | tee -a "$LOG_FILE"
echo "   - After Encoder: 9 experiments (3 noise types × 3 sampling)" | tee -a "$LOG_FILE"
echo "   - After Projector: 9 experiments (3 noise types × 3 sampling)" | tee -a "$LOG_FILE"
echo "   - After MFF_OCE: 9 experiments (3 noise types × 3 sampling)" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "4. Sampling Method Variations:" | tee -a "$LOG_FILE"
echo "   - Greedy Sampling: 9 experiments (3 noise types × 3 positions)" | tee -a "$LOG_FILE"
echo "   - Random Sampling: 9 experiments (3 noise types × 3 positions)" | tee -a "$LOG_FILE"
echo "   - K-means Sampling: 9 experiments (3 noise types × 3 positions)" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

echo "Check individual experiment logs in: $RESULTS_DIR/exp_*.log" | tee -a "$LOG_FILE"
