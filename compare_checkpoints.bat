@echo off
REM Compare two codec checkpoints using PSNR evaluation
REM checkpoint-30000 vs ckpt_25000

setlocal enableextensions enabledelayedexpansion

cd /d "%~dp0"
echo.
echo ============================================================
echo  Codec Checkpoint Comparison
echo ============================================================
echo.

set MIRA_VENV=..\mira\.venv\Scripts\python.exe
set CKPT_30K=outputs\codec_checkpoint-30000\checkpoint.pth
set CKPT_25K=outputs\codec_ckpt_25000.pth
set DATA_INDEX=C:\recordings\mira_wds\train\index.json
set NUM_CLIPS=16

if not exist "%CKPT_30K%" (
    echo ERROR: %CKPT_30K% not found
    exit /b 1
)

if not exist "%CKPT_25K%" (
    echo ERROR: %CKPT_25K% not found
    exit /b 1
)

if not exist "%DATA_INDEX%" (
    echo ERROR: Data index not found: %DATA_INDEX%
    exit /b 1
)

echo Evaluating checkpoint-30000...
echo.
"%MIRA_VENV%" codec_recon_psnr.py --codec "%CKPT_30K%" --data "%DATA_INDEX%" --num %NUM_CLIPS% > temp_30k.txt 2>&1

echo.
echo Evaluating ckpt_25000...
echo.
"%MIRA_VENV%" codec_recon_psnr.py --codec "%CKPT_25K%" --data "%DATA_INDEX%" --num %NUM_CLIPS% > temp_25k.txt 2>&1

echo.
echo ============================================================
echo Results Summary
echo ============================================================
echo.

echo [checkpoint-30000]:
findstr "mean" temp_30k.txt
if !ERRORLEVEL! neq 0 findstr /c:"PSNR" temp_30k.txt | tail -1

echo.
echo [ckpt_25000]:
findstr "mean" temp_25k.txt
if !ERRORLEVEL! neq 0 findstr /c:"PSNR" temp_25k.txt | tail -1

echo.
echo Full output saved to:
echo   temp_30k.txt  (checkpoint-30000)
echo   temp_25k.txt  (ckpt_25000)
echo.

del temp_30k.txt temp_25k.txt 2>nul

endlocal
