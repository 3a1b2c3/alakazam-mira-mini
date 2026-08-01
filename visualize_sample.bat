@echo off
REM 3D visualize a sample from MIRA WebDataset.
REM Usage: visualize_sample.bat [--sample 0] [--output out.png]
setlocal
python "%~dp0visualize_sample.py" --data C:\recordings\mira_wds\train\index.json %*
endlocal
