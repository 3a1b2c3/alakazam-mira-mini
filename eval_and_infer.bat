@echo off
REM Evaluate and infer on world model checkpoints
REM Usage: eval_and_infer.bat [checkpoint_path]

setlocal enabledelayedexpansion

if "%1"=="" (
    set CKPT=outputs\scratch_ch\checkpoint-56000\checkpoint-21000-horde.pth
) else (
    set CKPT=%1
)

echo ============================================================
echo World Model Evaluation and Inference
echo ============================================================
echo Checkpoint: !CKPT!
echo.

call .\.venv\Scripts\activate.bat

if "%2"=="infer" (
    echo [1/1] Running inference...
    python infer_world_model.py
    goto :end
)

echo [1/2] Inspecting checkpoint...
python inspect_checkpoint.py
echo.

echo [2/2] Running inference...
python infer_world_model.py

:end
echo.
echo ============================================================
echo Done. Output in outputs/wm_inference/
echo ============================================================
