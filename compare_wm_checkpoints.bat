@echo off
REM Compare world-model checkpoints (70k vs 98k) using eval_wm.bat
setlocal enableextensions enabledelayedexpansion

cd /d "%~dp0"
echo.
echo ============================================================
echo  World-Model Checkpoint Comparison (70k vs 98k)
echo ============================================================
echo.

set "CKPT_70K=%cd%\outputs\wm_ckpt_70000.pth"
set "CKPT_98K=%cd%\outputs\wm_ckpt_98000.pth"

if not exist "%CKPT_70K%" (
    echo ERROR: %CKPT_70K% not found
    exit /b 1
)

if not exist "%CKPT_98K%" (
    echo ERROR: %CKPT_98K% not found
    exit /b 1
)

echo [1/2] Evaluating wm_ckpt_70000...
echo.
call eval_wm.bat "%CKPT_70K%" --num-samples 16 --val-n-samples 8 > "%temp%\eval_70k.txt" 2>&1
type "%temp%\eval_70k.txt"
echo.

echo.
echo [2/2] Evaluating wm_ckpt_98000...
echo.
call eval_wm.bat "%CKPT_98K%" --num-samples 16 --val-n-samples 8 > "%temp%\eval_98k.txt" 2>&1
type "%temp%\eval_98k.txt"
echo.

echo ============================================================
echo  Results Summary
echo ============================================================
echo.
echo [wm_ckpt_70000]:
findstr /E "val_loss gFDD gFID PSNR latent mean" "%temp%\eval_70k.txt" | findstr /V "INFO DEBUG"
echo.
echo [wm_ckpt_98000]:
findstr /E "val_loss gFDD gFID PSNR latent mean" "%temp%\eval_98k.txt" | findstr /V "INFO DEBUG"
echo.

echo Full output saved to:
echo   %temp%\eval_70k.txt
echo   %temp%\eval_98k.txt
echo.

endlocal
