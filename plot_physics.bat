@echo off
REM Plot physics data (position, velocity, rotation) from RacerX dataset.
REM Usage: plot_physics.bat [--sample 0] [--output out.png]
setlocal
python "%~dp0plot_physics.py" --data C:\recordings\mira_wds\train\index.json %*
endlocal
