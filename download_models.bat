@echo off
setlocal enableextensions
REM ==========================================================================
REM Download MIRA data/models. Requires setup.bat first (needs the .venv + mira[hf]).
REM   - rocket-science DATASET (HuggingFace kyutai/rocket-science): the 2v2 clips +
REM     actions + game state. This is the main download; no pretrained MIRA checkpoints
REM     are published (train from scratch with scripts/train_*.py).
REM   - DINOv3-L/16 (codec TRAINING only, gated by Meta) -- see note at the bottom.
REM
REM Knobs:  set SPLIT=test|train   set SHARDS=1  (1 = one shard/quick; 0 = full split)
REM Usage:  download_models.bat
REM ==========================================================================
set "MIRA=%~dp0..\mira"
set "PY=%MIRA%\.venv\Scripts\python.exe"
if not exist "%PY%" ( echo ERROR: .venv missing -- run setup.bat first & exit /b 1 )
if not defined SPLIT set "SPLIT=test"
if not defined SHARDS set "SHARDS=1"
REM py3.11 venv, but keep HF single-stream/robust on Windows anyway
set "HF_HUB_ENABLE_HF_TRANSFER=0"
set "HF_HUB_DISABLE_XET=1"

echo --- downloading rocket-science: split=%SPLIT% shards=%SHARDS% (0=full) ---
"%PY%" -c "from mira.data import RocketScienceDataset; ds=RocketScienceDataset.from_hub('kyutai/rocket-science', split='%SPLIT%', shards=(%SHARDS% or None)); print('downloaded', len(ds.match_ids()), 'matches for split %SPLIT%')"
if errorlevel 1 ( echo ERROR: dataset download failed & exit /b 1 )

echo.
echo === done: rocket-science (%SPLIT%) cached ===
echo.
echo NOTE: DINOv3-L/16 is only needed for CODEC TRAINING (not world-model train/inference).
echo It is gated by Meta (dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth). If you have access,
echo place it in a folder and:  set RS_DINO_WEIGHTS_DIR=C:\path\to\weights
echo (facebook/dinov3-vitl16-pretrain-lvd1689m already appears in your HuggingFace cache.)
endlocal
