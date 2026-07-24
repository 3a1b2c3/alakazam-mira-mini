@echo off
setlocal enableextensions
REM ==========================================================================
REM Pre-download MIRA Mini weights so `mira-mini play` starts with no wait.
REM Reuses the WorldCanvas .venv (where alakazam-mira-mini + huggingface_hub are
REM installed). Uses hf_xet (faster); set HF_HUB_DISABLE_XET=1 first if it hangs.
REM Usage:  download_weights.bat [1b|364m|all|<repo_id>]   (default: 1b)
REM ==========================================================================
set "HERE=%~dp0"
set "PY=C:\workspace\world\WorldCanvas\.venv\Scripts\python.exe"
if not exist "%PY%" ( echo ERROR: WorldCanvas .venv python not found at %PY% & exit /b 1 )

set "WHICH=%~1"
if "%WHICH%"=="" set "WHICH=1b"

echo GPU (weights are large; make sure the disk has room):
nvidia-smi --query-gpu=memory.free --format=csv,noheader
echo.
"%PY%" "%HERE%download_weights.py" %WHICH%
endlocal
