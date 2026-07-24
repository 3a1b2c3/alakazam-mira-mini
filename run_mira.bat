@echo off
setlocal enableextensions
REM ==========================================================================
REM Launch MIRA Mini locally in the browser. Uses the WorldCanvas .venv where
REM alakazam-mira-mini is installed. First run downloads the weights on demand
REM (or run download_weights.bat first). --model auto picks the 1b on CUDA.
REM Usage:  run_mira.bat [extra args]   e.g.  run_mira.bat --model 364m --steps 8
REM ==========================================================================
set "MIRA=C:\workspace\world\WorldCanvas\.venv\Scripts\mira-mini.exe"
if not exist "%MIRA%" ( echo ERROR: mira-mini not found at %MIRA% - install alakazam-mira-mini into that venv first & exit /b 1 )

echo GPU before launch:
nvidia-smi --query-gpu=memory.total,memory.used,memory.free --format=csv,noheader
echo.
echo Starting MIRA Mini play - a browser tab will open when it is ready...
REM Default to the 1b model (this box has a discrete GPU). A later --model on the
REM command line overrides it, e.g. run_mira.bat --model 364m
"%MIRA%" play --model 1b %*
endlocal
