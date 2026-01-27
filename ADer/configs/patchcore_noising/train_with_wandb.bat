@echo off
REM Training script with WandB for PatchCore Noising (Reverse Distillation)

REM Set your WandB entity (username or team)
SET WANDB_ENTITY=

REM Dataset settings
SET DATA_ROOT=data/mvtec
SET CLS_NAMES=bottle

REM Training settings
SET CONFIG=configs.patchcore_noising.patchcore_noising_256_100e
SET EPOCHS=100
SET BATCH_SIZE=8
SET LR=0.005

REM WandB settings
SET WANDB_PROJECT=patchcore-rd-anomaly
SET RUN_NAME=patchcore_rd_%CLS_NAMES%_%EPOCHS%e

echo ======================================
echo PatchCore RD Training with WandB
echo ======================================
echo Config: %CONFIG%
echo Dataset: %DATA_ROOT%
echo Classes: %CLS_NAMES%
echo Epochs: %EPOCHS%
echo WandB Project: %WANDB_PROJECT%
echo Run Name: %RUN_NAME%
echo ======================================

REM Run training
python main.py ^
    --config %CONFIG% ^
    --cls_names %CLS_NAMES% ^
    --data.root %DATA_ROOT% ^
    --epoch_full %EPOCHS% ^
    --batch_train %BATCH_SIZE% ^
    --lr %LR% ^
    --wandb.enable True ^
    --wandb.project %WANDB_PROJECT% ^
    --wandb.name %RUN_NAME% ^
    --wandb.entity %WANDB_ENTITY% ^
    --wandb.log_model True

echo ======================================
echo Training completed!
echo Check WandB dashboard for results
echo ======================================
pause
