@echo off
setlocal enabledelayedexpansion

set "CHECKPOINT=%1"
set "STEP=%2"

if "%CHECKPOINT%"=="" (
    echo Usage: evaluate_all.bat checkpoint step
    echo Example: evaluate_all.bat outputs/wm_ckpt_63000.pth 63000
    exit /b 1
)

echo ============================================================
echo Evaluate Multiple Videos - Checkpoint: %CHECKPOINT%
echo ============================================================

echo.
echo === VIDEO 1: c00000 ===
call .\generate_and_eval.bat "%CHECKPOINT%" "%STEP%" "C:\recordings\mira_wds\test\000\dataset_00000\1ea1a4ce-2fdc-44fe-afdd-8019fbacea28_clip00000_c00000.p0.mp4" 24 c00000

echo.
echo === VIDEO 2: c00001 ===
call .\generate_and_eval.bat "%CHECKPOINT%" "%STEP%" "C:\recordings\mira_wds\test\000\dataset_00000\1ea1a4ce-2fdc-44fe-afdd-8019fbacea28_clip00000_c00001.p0.mp4" 20 c00001

echo.
echo === COMPLETE ===

