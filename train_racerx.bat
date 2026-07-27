@echo off
setlocal enableextensions
REM ==========================================================================
REM Train the RacerX world model FROM SCRATCH on the frozen mira-mini codec,
REM with honest eval on the held-out test split. Nothing hardcoded: data root is
REM derived relative to this bat; the codec is globbed from the HF cache.
REM Data location is a PARAMETER: set RX_ROOT to the mira_wds dir (holds train/ + test/);
REM defaults to <this repo>\..\..\..\recordings\mira_wds (relative to this bat).
REM Usage:
REM   train_racerx.bat                  20000 steps, default data root
REM   train_racerx.bat 50000
REM   set RX_ROOT=D:\data\mira_wds  &  train_racerx.bat 20000 run.batch_size=2
REM ==========================================================================
set "MIRA=%~dp0..\mira"
if defined RX_ROOT ( set "REC=%RX_ROOT%" ) else ( for %%I in ("%~dp0..\..\..\recordings\mira_wds") do set "REC=%%~fI" )
set "RECF=%REC:\=/%"

if not exist "%REC%\train\index.json" ( echo ERROR: no RacerX train index at %REC%\train -- build it: make_mira.bat all & exit /b 1 )
set "TESTIDX=%RECF%/train/index.json"
if exist "%REC%\test\index.json" ( set "TESTIDX=%RECF%/test/index.json" & echo eval on held-out test split ) else ( echo NOTE: no test split -- eval reuses train ^(run holdout_wds.py^) )

REM frozen codec from the mira-mini HF snapshot (globbed -> no hardcoded snapshot hash)
set "CODEC="
for /d %%S in ("%USERPROFILE%\.cache\huggingface\hub\models--alakazamworld--mira-mini\snapshots\*") do if exist "%%S\codec\checkpoint-125000\checkpoint.pth" set "CODEC=%%S\codec\checkpoint-125000\checkpoint.pth"
if not defined CODEC ( echo ERROR: mira-mini codec not found -- run download_weights.bat 1b & exit /b 1 )
set "CODEC=%CODEC:\=/%"

set "STEPS=%~1"
if "%STEPS%"=="" set "STEPS=20000"

REM single-GPU: clear stray torchrun env (Windows has no NCCL)
set "LOCAL_RANK=" & set "RANK=" & set "WORLD_SIZE=" & set "MASTER_ADDR=" & set "MASTER_PORT="
cd /d "%MIRA%"
echo GPU free:
nvidia-smi --query-gpu=memory.free --format=csv,noheader
echo codec  = %CODEC%
echo train  = %RECF%/train/index.json
echo test   = %TESTIDX%
echo output = %RECF%/wm_racerx  (%STEPS% steps)
echo.
"%MIRA%\.venv\Scripts\python.exe" scripts\train_world_model.py model.architecture.config.codec_checkpoint="%CODEC%" dataset.train_index=%RECF%/train/index.json dataset.test_index=%TESTIDX% run.batch_size=1 run.compile=false wandb.mode=offline dataloader.num_workers=0 run.steps=%STEPS% run.output_dir=%RECF%/wm_racerx %2 %3 %4 %5 %6
endlocal
