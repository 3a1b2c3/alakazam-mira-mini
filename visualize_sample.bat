@echo off
REM 3D visualize a sample from MIRA WebDataset.
REM Usage: visualize_sample.bat [--sample 0] [--output out.png]
REM Example: visualize_sample.bat --sample 5 --output sample5.png
setlocal
if "%~1"=="" (
    python "%~dp0visualize_sample.py" --data C:\recordings\mira_wds\train\index.json
) else (
    python "%~dp0visualize_sample.py" --data C:\recordings\mira_wds\train\index.json %*
)
endlocal
