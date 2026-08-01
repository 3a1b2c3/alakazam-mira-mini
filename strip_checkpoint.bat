@echo off
REM Strip optimizer state from checkpoint (29 GB -> ~5-10 GB for inference/archival).
REM Usage: strip_checkpoint.bat <checkpoint-dir> [--output <out.pth>]
setlocal
python "%~dp0strip_checkpoint.py" %*
endlocal
