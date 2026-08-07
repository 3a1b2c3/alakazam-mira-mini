@echo off
REM Evaluate codec checkpoint PSNR on RacerX clips
REM Uses the mira venv (has torchcodec + mira)

cd /d "%~dp0"
echo.
echo ============================================================
echo  Codec Checkpoint PSNR Evaluation (codec_checkpoint-30000)
echo ============================================================
echo.

set MIRA_VENV=..\mira\.venv\Scripts\python.exe
set CODEC_CKPT=outputs\codec_checkpoint-30000\checkpoint.pth
set DATA_INDEX=C:\recordings\mira_wds\train\index.json

if not exist "%CODEC_CKPT%" (
    echo ERROR: Checkpoint not found: %CODEC_CKPT%
    exit /b 1
)

if not exist "%DATA_INDEX%" (
    echo ERROR: Data index not found: %DATA_INDEX%
    exit /b 1
)

echo Checkpoint: %CODEC_CKPT%
echo Data index: %DATA_INDEX%
echo.

echo [1/1] Running codec evaluation...
"%MIRA_VENV%" codec_recon_psnr.py --codec "%CODEC_CKPT%" --data "%DATA_INDEX%" --num 16
set EXIT_CODE=%ERRORLEVEL%

echo.
echo ============================================================
if %EXIT_CODE% equ 0 (
    echo Complete
) else (
    echo ERROR: Evaluation failed with code %EXIT_CODE%
)
echo ============================================================
echo.

exit /b %EXIT_CODE%
