#!/bin/bash

# ============================================================================
# RDPP Noising - Sampling Ratio Experiments
# ============================================================================
# Base config: gaussian noise, encoder position, kmeans sampling
# Vary coreset_sampling_ratio: 1%, 5%, 10%, 20%, 50%, 100%
#
# Usage: ./run_rdpp_sampling_ratio.sh [gpu_id]
#
# Examples:
#   ./run_rdpp_sampling_ratio.sh        # Default GPU 0
#   ./run_rdpp_sampling_ratio.sh 1      # Use GPU 1
# ============================================================================

set -e

# Configuration
CONFIG_FILE="configs/rdpp_noising/rdpp_noising_256_100e_realiad.py"
GPU_ID=${1:-0}

# Fixed experiment settings (base)
NOISE_TYPE="gaussian"
NOISE_POSITION="encoder"
SAMPLING_METHOD="kmeans"

# Sampling ratios to test
SAMPLING_RATIOS=("0.01" "0.05" "0.10" "0.20" "0.50" "1.0")
SAMPLING_LABELS=("1pct" "5pct" "10pct" "20pct" "50pct" "100pct")

# Create results directory
RESULTS_DIR="results/rdpp_sampling_ratio_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_DIR"

# Log file
LOG_FILE="$RESULTS_DIR/experiments.log"

TOTAL_EXPS=${#SAMPLING_RATIOS[@]}
CURRENT_EXP=0

echo "============================================================================" | tee -a "$LOG_FILE"
echo "RDPP Noising - Sampling Ratio Experiments" | tee -a "$LOG_FILE"
echo "============================================================================" | tee -a "$LOG_FILE"
echo "Base Config:" | tee -a "$LOG_FILE"
echo "  Noise Type:     $NOISE_TYPE" | tee -a "$LOG_FILE"
echo "  Noise Position: $NOISE_POSITION" | tee -a "$LOG_FILE"
echo "  Sampling Method: $SAMPLING_METHOD" | tee -a "$LOG_FILE"
echo "  GPU: $GPU_ID" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "Sampling Ratios: ${SAMPLING_RATIOS[*]}" | tee -a "$LOG_FILE"
echo "Total Experiments: $TOTAL_EXPS" | tee -a "$LOG_FILE"
echo "Started at: $(date)" | tee -a "$LOG_FILE"
echo "============================================================================" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Loop through all sampling ratios
for i in "${!SAMPLING_RATIOS[@]}"; do
    ratio="${SAMPLING_RATIOS[$i]}"
    label="${SAMPLING_LABELS[$i]}"
    CURRENT_EXP=$((CURRENT_EXP + 1))

    EXP_NAME="${NOISE_TYPE}_${NOISE_POSITION}_${SAMPLING_METHOD}_${label}"

    echo "============================================================================" | tee -a "$LOG_FILE"
    echo "[$CURRENT_EXP/$TOTAL_EXPS] Running: $EXP_NAME" | tee -a "$LOG_FILE"
    echo "  Noise Type:            $NOISE_TYPE" | tee -a "$LOG_FILE"
    echo "  Noise Position:        $NOISE_POSITION" | tee -a "$LOG_FILE"
    echo "  Sampling Method:       $SAMPLING_METHOD" | tee -a "$LOG_FILE"
    echo "  Coreset Sampling Ratio: $ratio ($label)" | tee -a "$LOG_FILE"
    echo "  Started at: $(date)" | tee -a "$LOG_FILE"

    CUDA_VISIBLE_DEVICES=$GPU_ID python run.py \
        -c "$CONFIG_FILE" \
        -m train \
        model.kwargs.enable_noise=True \
        model.kwargs.noise_type="$NOISE_TYPE" \
        model.kwargs.noise_position="$NOISE_POSITION" \
        model.kwargs.coreset_sampling_ratio="$ratio" \
        trainer.sampling_method="$SAMPLING_METHOD" \
        trainer.noise_enabled=True \
        trainer.logdir_sub="$EXP_NAME" \
        wandb.name="$EXP_NAME" \
        wandb.tags="['rdpp','noising','$NOISE_TYPE','$NOISE_POSITION','$SAMPLING_METHOD','sampling_ratio','$label']" \
        2>&1 | tee "$RESULTS_DIR/exp_${CURRENT_EXP}_${EXP_NAME}.log"

    echo "  Completed at: $(date)" | tee -a "$LOG_FILE"
    echo "" | tee -a "$LOG_FILE"
done

# ============================================================================
# Summary
# ============================================================================
echo "============================================================================" | tee -a "$LOG_FILE"
echo "All Sampling Ratio Experiments Completed!" | tee -a "$LOG_FILE"
echo "Total experiments run: $TOTAL_EXPS" | tee -a "$LOG_FILE"
echo "Finished at: $(date)" | tee -a "$LOG_FILE"
echo "Results saved in: $RESULTS_DIR" | tee -a "$LOG_FILE"
echo "============================================================================" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "Experiment Summary:" | tee -a "$LOG_FILE"
echo "==================" | tee -a "$LOG_FILE"
echo "Base: gaussian + encoder + kmeans" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "  1.  1% sampling (coreset_sampling_ratio=0.01)" | tee -a "$LOG_FILE"
echo "  2.  5% sampling (coreset_sampling_ratio=0.05)" | tee -a "$LOG_FILE"
echo "  3. 10% sampling (coreset_sampling_ratio=0.10)" | tee -a "$LOG_FILE"
echo "  4. 20% sampling (coreset_sampling_ratio=0.20)" | tee -a "$LOG_FILE"
echo "  5. 50% sampling (coreset_sampling_ratio=0.50)" | tee -a "$LOG_FILE"
echo "  6. 100% sampling (coreset_sampling_ratio=1.0) - no subsampling" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "Check individual experiment logs in: $RESULTS_DIR/exp_*.log" | tee -a "$LOG_FILE"
