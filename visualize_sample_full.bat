@echo off
REM Combined visualization: video frames + actions + physics.
REM Usage: visualize_sample_full.bat [--sample 0] [--output out.png]
setlocal
python "%~dp0visualize_sample_full.py" --data C:\recordings\mira_wds\train\index.json %*
endlocal
