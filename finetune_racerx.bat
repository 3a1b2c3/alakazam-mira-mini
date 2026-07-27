@echo off
setlocal enableextensions
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

REM single-GPU: clear stray torchrun env (Windows has no NCCL)
set "LOCAL_RANK=" & set "RANK=" & set "WORLD_SIZE=" & set "MASTER_ADDR=" & set "MASTER_PORT="
cd /d "%MIRA%"
echo GPU free:
nvidia-smi --query-gpu=memory.free --format=csv,noheader
echo codec      = %CODEC%
echo warm-start = %WM%
echo train      = %RECF%/train/index.json
echo test       = %TESTIDX%
echo output     = %RECF%/wm_racerx_ft  (%STEPS% steps)
echo.
"%MIRA%\.venv\Scripts\python.exe" scripts\train_world_model.py model.architecture.config.codec_checkpoint="%CODEC%" run.finetune_from="%WM%" dataset.train_index=%RECF%/train/index.json dataset.test_index=%TESTIDX% run.batch_size=1 run.compile=false wandb.mode=offline dataloader.num_workers=0 run.steps=%STEPS% run.output_dir=%RECF%/wm_racerx_ft %2 %3 %4 %5 %6
endlocal
