@echo off
REM ============================================================================
REM RDPP Noising Single Experiment Runner (Windows)
REM ============================================================================
REM Usage: run_rdpp_single.bat <noise_type> <noise_position> <sampling_method>
REM
REM Examples:
REM   run_rdpp_single.bat uniform encoder greedy
REM   run_rdpp_single.bat gaussian projector random
REM   run_rdpp_single.bat perlin mff_oce kmeans
REM   run_rdpp_single.bat none none none
REM ============================================================================

setlocal

REM Configuration
set CONFIG_FILE=configs/rdpp_noising/rdpp_noising_256_100e.py
set GPU_ID=0

REM Parse arguments with defaults
set NOISE_TYPE=%1
set NOISE_POSITION=%2
set SAMPLING_METHOD=%3

if "%NOISE_TYPE%"=="" set NOISE_TYPE=uniform
if "%NOISE_POSITION%"=="" set NOISE_POSITION=encoder
if "%SAMPLING_METHOD%"=="" set SAMPLING_METHOD=greedy

REM Check if no noise mode
if "%NOISE_TYPE%"=="none" (
    set ENABLE_NOISE=False
    set EXP_NAME=no_noise_baseline
    echo ============================================================================
    echo Running RDPP Noising Experiment: No Noise ^(Baseline^)
    echo ============================================================================
) else (
    set ENABLE_NOISE=True
    set EXP_NAME=%NOISE_TYPE%_%NOISE_POSITION%_%SAMPLING_METHOD%
    echo ============================================================================
    echo Running RDPP Noising Experiment
    echo ============================================================================
    echo Noise Type: %NOISE_TYPE%
    echo Noise Position: %NOISE_POSITION%
    echo Sampling Method: %SAMPLING_METHOD%
)

echo Experiment Name: %EXP_NAME%
echo Started at: %date% %time%
echo ============================================================================
echo.

REM Set GPU
set CUDA_VISIBLE_DEVICES=%GPU_ID%

REM Run experiment
if "%ENABLE_NOISE%"=="False" (
    python run.py -c %CONFIG_FILE% -m train model.kwargs.enable_noise=False trainer.logdir_sub="%EXP_NAME%"
) else (
    python run.py -c %CONFIG_FILE% -m train model.kwargs.enable_noise=True model.kwargs.noise_type="%NOISE_TYPE%" model.kwargs.noise_position="%NOISE_POSITION%" trainer.sampling_method="%SAMPLING_METHOD%" trainer.logdir_sub="%EXP_NAME%"
)

echo.
echo ============================================================================
echo Experiment Completed!
echo Finished at: %date% %time%
echo ============================================================================

endlocal
