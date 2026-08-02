@echo off
REM Visualize a MIRA WebDataset sample: video frames + key presses + physics.
REM Usage: visualize_sample.bat [--sample 0] [--output out.png]
REM Example: visualize_sample.bat --sample 5 --output sample5.png
setlocal
set "PY=%~dp0.venv\Scripts\python.exe"
if not exist "%PY%" set "PY=python"
"%PY%" "%~dp0visualize_sample.py" --data C:\recordings\mira_wds\train\index.json %*
endlocal
