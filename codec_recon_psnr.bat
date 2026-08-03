@echo off
setlocal enableextensions
REM Codec encode->decode reconstruction PSNR on RacerX clips -- isolates STAGE 1
REM (no world model). This is the CEILING on video quality: the world model can
REM never render better RacerX than the codec reconstructs.
REM   << ~28.6  -> the (Rocket League) codec can't hold RacerX -> stage-1 IS the
REM               domain-transfer bottleneck (finetune/retrain the codec).
REM   ~ target  -> codec is fine; look at the WM finetune / data instead.
REM Uses the sibling mira venv + local DINO weights (so it never downloads DINO).
REM Usage:
REM   codec_recon_psnr.bat                 cuda, 8 clips
REM   codec_recon_psnr.bat --num 16
REM   codec_recon_psnr.bat --device cpu    (if the GPU is busy)
set "MIRA=%~dp0..\mira"
set "PY=%MIRA%\.venv\Scripts\python.exe"
if not exist "%PY%" ( echo ERROR: mira venv not found at %PY% & exit /b 1 )
REM point the codec's frozen DINOv3 at local weights (avoids the broken torch.hub download)
set "RS_DINO_WEIGHTS_DIR=%MIRA%\dino_weights"
REM clear stray torchrun env so it runs single-process
set "LOCAL_RANK="
set "RANK="
set "WORLD_SIZE="
set "MASTER_ADDR="
set "MASTER_PORT="
"%PY%" "%~dp0codec_recon_psnr.py" --data C:\recordings\mira_wds\train\index.json %*
endlocal
