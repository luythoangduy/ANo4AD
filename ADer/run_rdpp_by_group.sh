#!/bin/bash

# ============================================================================
# RDPP Noising Grouped Experiments Runner
# ============================================================================
# Usage: ./run_rdpp_by_group.sh <group_type>
#
# Groups:
#   noise_types    - Compare all noise types (uniform, gaussian, perlin) with same position & sampling
#   positions      - Compare all positions (encoder, projector, mff_oce) with same noise & sampling
#   sampling       - Compare all sampling methods (greedy, random, kmeans) with same noise & position
#   baseline       - Run baseline (no noise) + best config
#
# Examples:
#   ./run_rdpp_by_group.sh noise_types
#   ./run_rdpp_by_group.sh positions
#   ./run_rdpp_by_group.sh sampling
#   ./run_rdpp_by_group.sh baseline
# ============================================================================

set -e

CONFIG_FILE="configs/rdpp_noising/rdpp_noising_256_100e.py"
GPU_ID=0
GROUP=${1:-"baseline"}

echo "============================================================================"
echo "RDPP Noising Grouped Experiments"
echo "Group: $GROUP"
echo "Started at: $(date)"
echo "============================================================================"
echo ""

case $GROUP in
    noise_types)
        echo "Running: Noise Type Comparison (encoder + greedy)"
        echo "Experiments: uniform, gaussian, perlin"
        echo ""

        # Uniform
        echo "[1/3] Uniform Noise"
        CUDA_VISIBLE_DEVICES=$GPU_ID python run.py \
            -c "$CONFIG_FILE" -m train \
            model.kwargs.enable_noise=True \
            model.kwargs.noise_type="uniform" \
            model.kwargs.noise_position="encoder" \
            trainer.sampling_method="greedy" \
            trainer.logdir_sub="uniform_encoder_greedy" \
            wandb.name="uniform_encoder_greedy"

        # Gaussian
        echo "[2/3] Gaussian Noise"
        CUDA_VISIBLE_DEVICES=$GPU_ID python run.py \
            -c "$CONFIG_FILE" -m train \
            model.kwargs.enable_noise=True \
            model.kwargs.noise_type="gaussian" \
            model.kwargs.noise_position="encoder" \
            trainer.sampling_method="greedy" \
            trainer.logdir_sub="gaussian_encoder_greedy" \
            wandb.name="gaussian_encoder_greedy"

        # Perlin
        echo "[3/3] Perlin Noise"
        CUDA_VISIBLE_DEVICES=$GPU_ID python run.py \
            -c "$CONFIG_FILE" -m train \
            model.kwargs.enable_noise=True \
            model.kwargs.noise_type="perlin" \
            model.kwargs.noise_position="encoder" \
            trainer.sampling_method="greedy" \
            trainer.logdir_sub="perlin_encoder_greedy" \
            wandb.name="perlin_encoder_greedy"
        ;;

    positions)
        echo "Running: Noise Position Comparison (uniform + greedy)"
        echo "Experiments: encoder, projector, mff_oce"
        echo ""

        # Encoder
        echo "[1/3] After Encoder"
        CUDA_VISIBLE_DEVICES=$GPU_ID python run.py \
            -c "$CONFIG_FILE" -m train \
            model.kwargs.enable_noise=True \
            model.kwargs.noise_type="uniform" \
            model.kwargs.noise_position="encoder" \
            trainer.sampling_method="greedy" \
            trainer.logdir_sub="uniform_encoder_greedy" \
            wandb.name="uniform_encoder_greedy"

        # Projector
        echo "[2/3] After Projector"
        CUDA_VISIBLE_DEVICES=$GPU_ID python run.py \
            -c "$CONFIG_FILE" -m train \
            model.kwargs.enable_noise=True \
            model.kwargs.noise_type="uniform" \
            model.kwargs.noise_position="projector" \
            trainer.sampling_method="greedy" \
            trainer.logdir_sub="uniform_projector_greedy" \
            wandb.name="uniform_projector_greedy"

        # MFF_OCE
        echo "[3/3] After MFF_OCE"
        CUDA_VISIBLE_DEVICES=$GPU_ID python run.py \
            -c "$CONFIG_FILE" -m train \
            model.kwargs.enable_noise=True \
            model.kwargs.noise_type="uniform" \
            model.kwargs.noise_position="mff_oce" \
            trainer.sampling_method="greedy" \
            trainer.logdir_sub="uniform_mff_oce_greedy" \
            wandb.name="uniform_mff_oce_greedy"
        ;;

    sampling)
        echo "Running: Sampling Method Comparison (uniform + encoder)"
        echo "Experiments: greedy, random, kmeans"
        echo ""

        # Greedy
        echo "[1/3] Greedy Sampling"
        CUDA_VISIBLE_DEVICES=$GPU_ID python run.py \
            -c "$CONFIG_FILE" -m train \
            model.kwargs.enable_noise=True \
            model.kwargs.noise_type="uniform" \
            model.kwargs.noise_position="encoder" \
            trainer.sampling_method="greedy" \
            trainer.logdir_sub="uniform_encoder_greedy" \
            wandb.name="uniform_encoder_greedy"

        # Random
        echo "[2/3] Random Sampling"
        CUDA_VISIBLE_DEVICES=$GPU_ID python run.py \
            -c "$CONFIG_FILE" -m train \
            model.kwargs.enable_noise=True \
            model.kwargs.noise_type="uniform" \
            model.kwargs.noise_position="encoder" \
            trainer.sampling_method="random" \
            trainer.logdir_sub="uniform_encoder_random" \
            wandb.name="uniform_encoder_random"

        # K-means
        echo "[3/3] K-means Sampling"
        CUDA_VISIBLE_DEVICES=$GPU_ID python run.py \
            -c "$CONFIG_FILE" -m train \
            model.kwargs.enable_noise=True \
            model.kwargs.noise_type="uniform" \
            model.kwargs.noise_position="encoder" \
            trainer.sampling_method="kmeans" \
            trainer.logdir_sub="uniform_encoder_kmeans" \
            wandb.name="uniform_encoder_kmeans"
        ;;

    baseline)
        echo "Running: Baseline Experiments"
        echo "Experiments: no_noise, uniform_encoder_greedy"
        echo ""

        # No noise
        echo "[1/2] No Noise (Baseline)"
        CUDA_VISIBLE_DEVICES=$GPU_ID python run.py \
            -c "$CONFIG_FILE" -m train \
            model.kwargs.enable_noise=False \
            trainer.logdir_sub="no_noise_baseline" \
            wandb.name="no_noise_baseline"

        # Best config (uniform + encoder + greedy)
        echo "[2/2] With Noise (uniform + encoder + greedy)"
        CUDA_VISIBLE_DEVICES=$GPU_ID python run.py \
            -c "$CONFIG_FILE" -m train \
            model.kwargs.enable_noise=True \
            model.kwargs.noise_type="uniform" \
            model.kwargs.noise_position="encoder" \
            trainer.sampling_method="greedy" \
            trainer.logdir_sub="uniform_encoder_greedy" \
            wandb.name="uniform_encoder_greedy"
        ;;

    *)
        echo "Error: Unknown group type '$GROUP'"
        echo ""
        echo "Available groups:"
        echo "  noise_types - Compare noise types (uniform, gaussian, perlin)"
        echo "  positions   - Compare noise positions (encoder, projector, mff_oce)"
        echo "  sampling    - Compare sampling methods (greedy, random, kmeans)"
        echo "  baseline    - Run baseline + best config"
        exit 1
        ;;
esac

echo ""
echo "============================================================================"
echo "Group Completed!"
echo "Finished at: $(date)"
echo "============================================================================"
