@echo off
setlocal enableextensions
set "HYDRA_FULL_ERROR=1"
REM ==========================================================================
REM Train the MIRA latent world model on the FROZEN mira-mini codec (125k).
REM Prereq: get_data.bat (writes data_paths.bat with TRAIN_INDEX/TEST_INDEX).
REM Single-5090 smoke defaults: single-player, batch 1, no compile, W&B offline.
REM Override any Hydra key by appending key=value, e.g.:
REM    train.bat run.steps=200 run.batch_size=1
REM For the full 1B: add   model/latent_world_model=1b
REM For 4-player:    add   model=multi_wrapper_world_model dataset.n_players=4
REM ==========================================================================
REM Launcher lives in alakazam-mira-mini; the trainer (.venv, scripts, configs) is the sibling
REM mira repo -- derived relative to this bat's folder, so no absolute path is hardcoded.
set "MIRA=%~dp0..\mira"
cd /d "%MIRA%"
set "PY=%MIRA%\.venv\Scripts\python.exe"

REM Frozen codec from the downloaded mira-mini bundle (forward slashes for Hydra).
set "CODEC_CKPT=C:/Users/kschmid/.cache/huggingface/hub/models--alakazamworld--mira-mini/snapshots/19d668ac39814e394ae4a8f698690f761facf437/codec/checkpoint-125000/checkpoint.pth"

if not exist "%MIRA%\data_paths.bat" ( echo ERROR: no data_paths.bat - run get_data.bat first & exit /b 1 )
call "%MIRA%\data_paths.bat"
if "%TRAIN_INDEX%"=="" ( echo ERROR: TRAIN_INDEX empty - re-run get_data.bat & exit /b 1 )

REM Force the SINGLE-PROCESS path: set_up_distributed() no-ops only when LOCAL_RANK
REM is unset. A stray LOCAL_RANK (from a prior torchrun) makes it call
REM init_process_group, which needs RANK and would use NCCL (not built on Windows).
REM Clear them so training runs plain single-GPU.
set "LOCAL_RANK="
set "RANK="
set "WORLD_SIZE="
set "MASTER_ADDR="
set "MASTER_PORT="

REM The DINOv3 codec encoder is internally torch.compile'd (inductor needs Triton).
REM triton-windows is installed, so compile runs. If it errors on a Triton/inductor
REM version mismatch, fall back to eager with:  set "TORCHDYNAMO_DISABLE=1"
set "WANDB_MODE=offline"
echo GPU free:
nvidia-smi --query-gpu=memory.free --format=csv,noheader
echo codec = %CODEC_CKPT%
echo train = %TRAIN_INDEX%
echo test  = %TEST_INDEX%
echo.

REM dataloader.num_workers=0 -> load in the main process: no Windows worker-spawn
REM overhead, and one shard doesn't need parallel readers. Raise it for real runs.
"%PY%" scripts/train_world_model.py model.architecture.config.codec_checkpoint="%CODEC_CKPT%" dataset.train_index="%TRAIN_INDEX%" dataset.test_index="%TEST_INDEX%" run.batch_size=1 run.compile=false wandb.mode=disabled dataloader.num_workers=0 ++tensorboard.logdir=${run.output_dir}/tb %*
endlocal
