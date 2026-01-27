#!/bin/bash

# Training script with WandB for PatchCore Noising (Reverse Distillation)

# Set your WandB entity (username or team)
WANDB_ENTITY=""  # Leave empty to use default

# Dataset settings
DATA_ROOT="data/mvtec"
CLS_NAMES="bottle"  # Change to your class name(s)

# Training settings
CONFIG="configs.patchcore_noising.patchcore_noising_256_100e"
EPOCHS=100
BATCH_SIZE=8
LR=0.005

# WandB settings
WANDB_PROJECT="patchcore-rd-anomaly"
RUN_NAME="patchcore_rd_${CLS_NAMES}_${EPOCHS}e"

echo "======================================"
echo "PatchCore RD Training with WandB"
echo "======================================"
echo "Config: $CONFIG"
echo "Dataset: $DATA_ROOT"
echo "Classes: $CLS_NAMES"
echo "Epochs: $EPOCHS"
echo "WandB Project: $WANDB_PROJECT"
echo "Run Name: $RUN_NAME"
echo "======================================"

# Run training
python main.py \
    --config $CONFIG \
    --cls_names $CLS_NAMES \
    --data.root $DATA_ROOT \
    --epoch_full $EPOCHS \
    --batch_train $BATCH_SIZE \
    --lr $LR \
    --wandb.enable True \
    --wandb.project $WANDB_PROJECT \
    --wandb.name $RUN_NAME \
    --wandb.entity $WANDB_ENTITY \
    --wandb.log_model True

echo "======================================"
echo "Training completed!"
echo "Check WandB dashboard for results"
echo "======================================"
