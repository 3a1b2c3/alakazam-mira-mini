@echo off
setlocal enableextensions
set "HYDRA_FULL_ERROR=1"
REM ==========================================================================
REM FINETUNE the RacerX world model by warm-starting from the mira-mini 1B
REM checkpoint (checkpoint-52000), on the frozen mira-mini codec, with honest
REM eval on the held-out test split. Recommended over from-scratch at limited data.
REM Nothing hardcoded: data root is a parameter; codec + WM are globbed from the HF cache.
REM
REM Data location is a PARAMETER: set RX_ROOT to the mira_wds dir (holds train/ + test/);
REM defaults to <this repo>\..\..\..\recordings\mira_wds (relative to this bat).
REM Usage:
REM   finetune_racerx.bat                 10000 steps, default data root
REM   finetune_racerx.bat 20000
REM   set RX_ROOT=D:\data\mira_wds  &  finetune_racerx.bat 10000 run.batch_size=2
REM ==========================================================================
set "MIRA=%~dp0..\mira"
if defined RX_ROOT ( set "REC=%RX_ROOT%" ) else ( for %%I in ("%~dp0..\..\..\recordings\mira_wds") do set "REC=%%~fI" )
set "RECF=%REC:\=/%"

if not exist "%REC%\train\index.json" ( echo ERROR: no RacerX train index at %REC%\train -- build it: make_mira.bat all & exit /b 1 )
set "TESTIDX=%RECF%/train/index.json"
if exist "%REC%\test\index.json" ( set "TESTIDX=%RECF%/test/index.json" & echo eval on held-out test split ) else ( echo NOTE: no test split -- eval reuses train ^(run holdout_wds.py^) )

REM frozen codec + warm-start WM from the mira-mini HF snapshot (globbed -> no hardcoded hash)
set "SNAP="
for /d %%S in ("%USERPROFILE%\.cache\huggingface\hub\models--alakazamworld--mira-mini\snapshots\*") do if exist "%%S\codec\checkpoint-125000\checkpoint.pth" set "SNAP=%%S"
if not defined SNAP ( echo ERROR: mira-mini weights not found -- run download_weights.bat 1b & exit /b 1 )
set "CODEC=%SNAP%\codec\checkpoint-125000\checkpoint.pth"
set "WM=%SNAP%\checkpoint-52000\checkpoint.pth"
if not exist "%WM%" ( echo ERROR: warm-start checkpoint-52000 not found in %SNAP% & exit /b 1 )
set "CODEC=%CODEC:\=/%"
set "WM=%WM:\=/%"

set "STEPS=%~1"
if "%STEPS%"=="" set "STEPS=10000"

REM AUTO-RESUME: if a checkpoint already exists in the output dir, CONTINUE from it (restores
REM optimizer + step counter -> picks up where a killed run left off); else warm-start from the
REM mira-mini WM. keep_recent=1 means at most one checkpoint dir exists.
set "OUTDIR=%REC%\wm_racerx_ft"
set "RESUME="
for /d %%D in ("%OUTDIR%\checkpoint-*") do if exist "%%D\checkpoint.pth" set "RESUME=%%D\checkpoint.pth"
if defined RESUME set "RESUME=%RESUME:\=/%"
if defined RESUME ( set "STARTARG=run.continue_from=%RESUME%" ) else ( set "STARTARG=run.finetune_from=%WM%" )

REM single-GPU: clear stray torchrun env (Windows has no NCCL)
set "LOCAL_RANK=" & set "RANK=" & set "WORLD_SIZE=" & set "MASTER_ADDR=" & set "MASTER_PORT="
cd /d "%MIRA%"
echo GPU free:
nvidia-smi --query-gpu=memory.free --format=csv,noheader
echo codec      = %CODEC%
echo start      = %STARTARG%
echo train      = %RECF%/train/index.json
echo test       = %TESTIDX%
echo output     = %RECF%/wm_racerx_ft  (%STEPS% steps)
echo.
"%MIRA%\.venv\Scripts\python.exe" scripts\train_world_model.py --config-name finetune_world_model model.architecture.config.codec_checkpoint="%CODEC%" %STARTARG% dataset.train_index=%RECF%/train/index.json dataset.test_index=%TESTIDX% run.batch_size=1 run.compile=true wandb.mode=disabled dataloader.num_workers=0 run.steps=%STEPS% run.output_dir=%RECF%/wm_racerx_ft run.checkpoint_every=250 optim.scheduler.warmup_steps=200 validation.val_n_samples=64 world_model_metrics.num_samples=128 ++tensorboard.logdir=%RECF%/wm_racerx_ft/tb %2 %3 %4 %5 %6
endlocal
