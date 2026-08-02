@echo off
REM Visualize a MIRA WebDataset sample: video frames + key presses + physics.
REM Usage: visualize_sample.bat [--start 0] [--count 1] [--output out.png]
REM   --start N         first sample index (default 0)
REM   --count M         render M samples from --start (out_<idx>.png each when >1)
REM Example: visualize_sample.bat --start 5 --count 3 --output C:\tmp\viz.png
setlocal
set "PY=%~dp0.venv\Scripts\python.exe"
if not exist "%PY%" set "PY=python"
"%PY%" "%~dp0visualize_sample.py" --data C:\recordings\mira_wds\train\index.json %*
endlocal
