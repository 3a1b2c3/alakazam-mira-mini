@echo off
setlocal enableextensions enabledelayedexpansion
REM ==========================================================================
REM Smoke-test EVERY runnable MIRA training path on the RacerX data, each capped
REM (few val samples, few steps) so it reaches real training steps + a checkpoint
REM in minutes. Reports PASS/FAIL by checking each run actually WROTE a checkpoint
REM -- exit code 0 alone isn't enough (a run can 0-exit after a stalled validation).
REM Runs all steps even if one fails, so a single failure doesn't hide the rest.
REM
REM NOT covered: SLURM (train.sbatch) needs a Linux/NCCL cluster; can't run here.
REM
REM Prereqs: free GPU (checked below); DINOv3-L/16 weights for the codec step;
REM data_paths.bat (get_data.bat) for the world-model steps; RacerX index at
REM C:\recordings\mira_wds\train\index.json.
REM
REM Usage:  smoke_test_all.bat           full suite (20 steps each)
REM         smoke_test_all.bat 10        override step count
REM ==========================================================================
REM Paths derived relative to this bat (in alakazam-mira-mini): the sibling mira repo and the
REM racer-x scripts dir -- no absolute repo paths hardcoded.
set "MIRA=%~dp0..\mira"
set "RX=%~dp0..\..\REMIX\racer-x\tools\replay-processing\scripts"
set "REC=C:\recordings\mira_wds"
set "RECF=C:/recordings/mira_wds"
set "IDX=C:/recordings/mira_wds/train/index.json"

set "STEPS=%~1"
if "%STEPS%"=="" set "STEPS=20"
set "CAPS=validation.val_n_samples=8 world_model_metrics.num_samples=16"

REM clear stray torchrun env so the single-GPU path is taken (Windows has no NCCL)
set "LOCAL_RANK=" & set "RANK=" & set "WORLD_SIZE=" & set "MASTER_ADDR=" & set "MASTER_PORT="

echo === GPU free (each training step needs a free GPU) ===
nvidia-smi --query-gpu=memory.free,memory.used --format=csv,noheader
echo.

set "R_DATA=?" & set "R_CODEC=?" & set "R_FT=?" & set "R_SCRATCH=?" & set "R_CHAIN=?"

echo ############### 1/5 data loader test (no GPU) ###############
call "%RX%\test_mira_dataset.bat"
if errorlevel 1 ( set "R_DATA=FAIL" ) else ( set "R_DATA=PASS" )

echo ############### 2/5 codec from scratch ###############
call "%RX%\train_mira_smoke.bat" %STEPS%
call :ckpt "%REC%\codec_smoke" R_CODEC

echo ############### 3/5 world model FINETUNE (warm start) ###############
call "%~dp0smoke_test.bat" run.steps=%STEPS% %CAPS% run.output_dir=%RECF%/wm_finetune_smoke
call :ckpt "%REC%\wm_finetune_smoke" R_FT

echo ############### 4/5 world model FROM SCRATCH ###############
call "%~dp0train.bat" run.steps=%STEPS% %CAPS% dataset.train_index=%IDX% dataset.test_index=%IDX% run.output_dir=%RECF%/wm_scratch_smoke
call :ckpt "%REC%\wm_scratch_smoke" R_SCRATCH

echo ############### 5/5 full chain (step-2 codec -^> world model) ###############
set "FRESH_CODEC="
for /d %%D in ("%REC%\codec_smoke\checkpoint-*") do if exist "%%D\checkpoint.pth" set "FRESH_CODEC=%%D\checkpoint.pth"
if not defined FRESH_CODEC (
    echo   SKIP: codec step produced no checkpoint to chain from
    set "R_CHAIN=SKIP"
) else (
    set "FC=!FRESH_CODEC:\=/!"
    "%MIRA%\.venv\Scripts\python.exe" "%MIRA%\scripts\train_world_model.py" model.architecture.config.codec_checkpoint="!FC!" dataset.train_index=%IDX% dataset.test_index=%IDX% run.batch_size=1 run.compile=false wandb.mode=offline dataloader.num_workers=0 run.steps=%STEPS% %CAPS% run.output_dir=%RECF%/wm_chain_smoke
    call :ckpt "%REC%\wm_chain_smoke" R_CHAIN
)

echo.
echo ============================================================
echo SMOKE TEST SUMMARY  (PASS = wrote a checkpoint)
echo   1 data loader test ......... !R_DATA!
echo   2 codec from scratch ....... !R_CODEC!
echo   3 world model finetune ..... !R_FT!
echo   4 world model from scratch . !R_SCRATCH!
echo   5 full chain (codec-^>WM) ... !R_CHAIN!
echo   - SLURM (train.sbatch) ..... N/A here (Linux/NCCL only)
echo ============================================================
endlocal
goto :eof

:ckpt
REM %1=output dir, %2=result var name. PASS iff a checkpoint-*\checkpoint.pth exists.
set "%~2=FAIL"
for /d %%D in ("%~1\checkpoint-*") do if exist "%%D\checkpoint.pth" set "%~2=PASS"
goto :eof
