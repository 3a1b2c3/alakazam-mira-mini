@echo off
setlocal enableextensions
set "HYDRA_FULL_ERROR=1"
REM ==========================================================================
REM Train MIRA stage 1 from scratch (no auto-resume).
REM Trains latent world model on FROZEN codec (125k).
REM
REM Usage:
REM    train_stage1.bat                       default: stage1_logs/
REM    train_stage1.bat run.steps=10000       custom steps
REM    train_stage1.bat run.batch_size=2      custom batch size
REM    train_stage1.bat model/latent_world_model=1b  full 1B model
REM ==========================================================================

set "MIRA=%~dp0..\mira"
cd /d "%MIRA%"
set "PY=%MIRA%\.venv\Scripts\python.exe"

REM Frozen codec from the downloaded mira-mini bundle (forward slashes for Hydra).
set "CODEC_CKPT=C:/Users/kschmid/.cache/huggingface/hub/models--alakazamworld--mira-mini/snapshots/19d668ac39814e394ae4a8f698690f761facf437/codec/checkpoint-125000/checkpoint.pth"

if not exist "%MIRA%\data_paths.bat" ( echo ERROR: no data_paths.bat - run get_data.bat first & exit /b 1 )
call "%MIRA%\data_paths.bat"
if "%TRAIN_INDEX%"=="" ( echo ERROR: TRAIN_INDEX empty - re-run get_data.bat & exit /b 1 )

REM Force the SINGLE-PROCESS path: no distributed training on Windows.
set "LOCAL_RANK="
set "RANK="
set "WORLD_SIZE="
set "MASTER_ADDR="
set "MASTER_PORT="

REM Stage 1: train from scratch (no auto-resume)
set "OUTPUT_DIR=stage1_logs"
set "CONTINUE_FROM="

set "WANDB_MODE=offline"
if not defined PYTORCH_CUDA_ALLOC_CONF set "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"

echo ============================================================
echo  Training Stage 1 (from scratch)
echo ============================================================
echo GPU free:
nvidia-smi --query-gpu=memory.free --format=csv,noheader
echo codec  = %CODEC_CKPT%
echo train  = %TRAIN_INDEX%
echo test   = %TEST_INDEX%
echo output = %OUTPUT_DIR%
echo.

REM Run training: stage 1 default is 1-player, batch 1, no compile
"%PY%" scripts/train_world_model.py ^
    model.architecture.config.codec_checkpoint="%CODEC_CKPT%" ^
    dataset.train_index="%TRAIN_INDEX%" ^
    dataset.test_index="%TEST_INDEX%" ^
    run.output_dir="%OUTPUT_DIR%" ^
    run.batch_size=1 ^
    run.compile=false ^
    wandb.mode=disabled ^
    dataloader.num_workers=0 ^
    run.log_every=50 ^
    validation.downstream_val_every=7000 ^
    run.checkpoint_every=7000 ^
    run.checkpoint_keep_recent=2 ^
    world_model_metrics.num_samples=32 ^
    world_model_metrics.dino_max_chunk_size=32 ^
    ++tensorboard.logdir=${run.output_dir}/tb ^
    %*

endlocal
