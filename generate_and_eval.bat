@echo off
REM Wrapper: Generate video from checkpoint and evaluate quality

setlocal enabledelayedexpansion

REM Defaults
set "CHECKPOINT=%1"
if "%CHECKPOINT%"=="" set "CHECKPOINT=outputs/wm_ckpt_63000.pth"

set "STEP=%2"
if "%STEP%"=="" set "STEP=63000"

set "VIDEO=%3"
if "%VIDEO%"=="" set "VIDEO=C:\recordings\mira_wds\test\000\dataset_00000\1ea1a4ce-2fdc-44fe-afdd-8019fbacea28_clip00000_c00000.p0.mp4"

set "GEN_FRAMES=%4"
if "%GEN_FRAMES%"=="" set "GEN_FRAMES=24"

set "CLIP_ID=%5"
if "%CLIP_ID%"=="" set "CLIP_ID=c00000"

REM Extract checkpoint name for output files
for %%F in (%CHECKPOINT%) do set "CKPT_NAME=%%~nF"
set "CKPT_NAME=!CKPT_NAME:.pth=!"

set "GEN_VIDEO=outputs\generated_native_!CKPT_NAME!_!CLIP_ID!.mp4"

echo.
echo ============================================================
echo Generate and Evaluate
echo ============================================================
echo.
echo Checkpoint: !CHECKPOINT!
echo Step:       !STEP!
echo Video:      !VIDEO!
echo Gen frames: !GEN_FRAMES!
echo Output:     !GEN_VIDEO!
echo.

REM Check actions exist
set "ACTION_FILE=!VIDEO:.mp4=.jsonl!"
if not exist "!ACTION_FILE!" (
    echo ERROR: Action file not found: !ACTION_FILE!
    exit /b 1
)

REM Step 1: Generate video
echo [1/2] Generating video...
.\.venv\Scripts\python.exe generate_from_inference.py "!CHECKPOINT!" --video "!VIDEO!" --output "!GEN_VIDEO!" --gen-frames !GEN_FRAMES!

if errorlevel 1 (
    echo ERROR: Generation failed
    exit /b 1
)

echo.
echo [2/2] Evaluating video quality...
.\.venv\Scripts\python.exe eval_video_quality.py --real "!VIDEO!" --gen "!GEN_VIDEO!" --tb-dir outputs/tb --step !STEP!

if errorlevel 1 (
    echo ERROR: Evaluation failed
    exit /b 1
)

echo.
echo ============================================================
echo Complete!
echo ============================================================
echo Video:   !GEN_VIDEO!
echo Metrics: outputs\tb
echo.

endlocal

