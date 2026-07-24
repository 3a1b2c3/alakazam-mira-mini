@echo off
setlocal enableextensions enabledelayedexpansion
REM ==========================================================================
REM Offline-evaluate a world-model checkpoint (metrics on a saved checkpoint,
REM no training). Runs mira's scripts/eval_world_model_offline.py, which reads
REM the run's world_model_config.yaml saved 2 dirs above the .pth, so the eval
REM matches how the model was trained. Single-process (no torchrun).
REM Launcher lives in alakazam-mira-mini; trainer/.venv is the sibling mira repo.
REM
REM Reports: validation loss + world-model metrics (DINO/latent drift, Frechet
REM DINO/Inception = gFDD/gFID-analogs). Capped small by default for speed --
REM raise --num-samples / --val-n-samples for a real (slow) eval.
REM
REM Usage:
REM   eval_wm.bat                          DEFAULT: newest local RacerX WM checkpoint
REM   eval_wm.bat <checkpoint.pth>         a specific checkpoint
REM   eval_wm.bat <ckpt> --num-samples 256 --val-n-samples 128     (fuller eval)
REM   eval_wm.bat <ckpt> --viz 4                                    (also render 4 rollouts)
REM   eval_wm.bat --viz 4                                           (default ckpt + extra args)
REM ==========================================================================
set "MIRA=%~dp0..\mira"
set "REC=C:\recordings\mira_wds"

REM First arg is the checkpoint unless it's a flag (starts with '-'); else use the default.
set "CKPT="
set "A1=%~1"
if defined A1 if not "%A1:~0,1%"=="-" (
    set "CKPT=%~1"
    shift
)

if not defined CKPT call :pickdefault
if not defined CKPT (
    echo ERROR: no WM checkpoint under %REC%\wm_* -- run a training first, or pass a path.
    exit /b 1
)
if not exist "%CKPT%" (
    echo ERROR: checkpoint not found: "%CKPT%"
    exit /b 1
)

REM collect any extra pass-through args
set "EXTRA="
:collect
if "%~1"=="" goto run
set "EXTRA=!EXTRA! %1"
shift
goto collect

:run
cd /d "%MIRA%"
set "PY=%MIRA%\.venv\Scripts\python.exe"
REM World-model metrics load DINOv3-base (dinov3_vitb16) via torch.hub for the DINO-Frechet
REM metric; point it at local weights. NOTE: dino_weights currently has only the LARGE
REM (vitl16); the base file dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth must be here too, or
REM the metrics crash on a Path(None) -- until then run this bat with --skip-metrics.
if "%RS_DINO_WEIGHTS_DIR%"=="" set "RS_DINO_WEIGHTS_DIR=%MIRA%\dino_weights"
REM single-GPU: clear stray torchrun env (Windows has no NCCL)
set "LOCAL_RANK="
set "RANK="
set "WORLD_SIZE="
set "MASTER_ADDR="
set "MASTER_PORT="

echo GPU free:
nvidia-smi --query-gpu=memory.free --format=csv,noheader
echo eval = %CKPT%
echo.

REM World-model metrics load DINOv3-base (vitb16) via torch.hub; the hub crashes on a
REM Path(None) if the weights aren't present. If the base file is missing, auto-skip the
REM metrics (validation-loss eval still runs) instead of crashing. Get the weights with
REM download_dino.bat (DINOV3_VITB16_URL) to enable full metrics.
set "VITB16=%RS_DINO_WEIGHTS_DIR%\dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"
set "AUTOSKIP="
if not exist "%VITB16%" (
    echo WARNING: DINOv3-base weights not found:
    echo          "%VITB16%"
    echo          world-model metrics need them -- running VALIDATION LOSS ONLY.
    echo          Run download_dino.bat ^(with DINOV3_VITB16_URL^) to enable full metrics.
    echo.
    set "AUTOSKIP=--skip-metrics"
)

REM Eval on the RacerX WebDataset (released checkpoints bake the author's absolute data paths
REM into their config; --test-index overrides that). Only add it when the index exists, so a
REM local checkpoint's own valid config still wins if RacerX isn't built. User --test-index in
REM the extra args overrides this (last wins).
set "TESTIDX=C:/recordings/mira_wds/train/index.json"
set "IDXARG="
if exist "C:\recordings\mira_wds\train\index.json" set "IDXARG=--test-index %TESTIDX%"

REM small caps by default so a smoke checkpoint evals in minutes; override by
REM passing your own --num-samples / --val-n-samples in the extra args (last wins).
"%PY%" scripts\eval_world_model_offline.py "%CKPT%" %IDXARG% --num-samples 16 --val-n-samples 8 %AUTOSKIP% %EXTRA%
endlocal
goto :eof

REM --- DEFAULT = newest WM checkpoint from a LOCAL RacerX run = highest checkpoint-<N> under
REM     %REC%\wm_* . NOTE: the released reference (checkpoint-52000) can NOT be the default --
REM     it's a Rocket League model with video.timesteps=80 (needs 160-frame clips), so it fails
REM     on RacerX's 80-frame chunks AND its numbers on RacerX would be domain-mismatched garbage.
REM     Only RacerX-trained checkpoints (timesteps=40 -> clip_len 80) eval on this data.
REM     'checkpoint-*' skips in-progress '.checkpoint-N.tmp' dirs; 'wm_*' skips the codec.
:pickdefault
set /a BEST=-1
for /d %%R in ("%REC%\wm_*") do for /d %%D in ("%%R\checkpoint-*") do call :consider "%%D"
if defined CKPT (
    echo default checkpoint ^(newest local RacerX WM^): "%CKPT%"
) else (
    echo ERROR: no RacerX WM checkpoint under %REC%\wm_* -- run a training first, or pass a path.
)
goto :eof

:consider
set "D=%~1"
if not exist "%D%\checkpoint.pth" goto :eof
set "N=%~nx1"
set "N=%N:checkpoint-=%"
REM skip non-numeric step names (guards the numeric compare below)
for /f "delims=0123456789" %%X in ("%N%") do goto :eof
if %N% GTR %BEST% (
    set /a BEST=%N%
    set "CKPT=%D%\checkpoint.pth"
)
goto :eof
