@echo off
setlocal enableextensions
REM ==========================================================================
REM Upload the racer-x mira WebDataset to a Horde instance over SSH (scp, using
REM Windows' built-in OpenSSH). Native-Windows alternative to upload_to_horde.sh
REM (which needs WSL rsync). NOTE: scp is NOT incremental -- it re-copies every
REM file each run; for repeated syncs use upload_to_horde.sh (rsync) from WSL.
REM Sends ONLY the shards (train\ + test\), not the wm_* / tb / wandb outputs.
REM
REM   set HOST=user@instance-ip           (required -- Horde instance SSH target)
REM   set DEST=/data/mira_wds             (default, remote path)
REM   set RX_ROOT=C:\recordings\mira_wds  (default, local data)
REM   set SSH_KEY=C:\path\to\key          (optional private key)
REM   upload_to_horde.bat
REM ==========================================================================
if "%HOST%"=="" ( echo ERROR: set HOST=user@instance-ip & exit /b 1 )
if not defined DEST set "DEST=/data/mira_wds"
if not defined RX_ROOT set "RX_ROOT=C:\recordings\mira_wds"
if not exist "%RX_ROOT%\train\index.json" ( echo ERROR: no data at %RX_ROOT%\train -- build the WebDataset first & exit /b 1 )
set "KEYOPT="
if defined SSH_KEY set KEYOPT=-i "%SSH_KEY%"

echo Creating %DEST% on %HOST% ...
ssh %KEYOPT% %HOST% "mkdir -p '%DEST%'"
if errorlevel 1 ( echo ssh failed -- check HOST / key / connectivity & exit /b 1 )

echo Copying %RX_ROOT%\train -^> %HOST%:%DEST%/ ...
scp %KEYOPT% -r "%RX_ROOT%\train" "%HOST%:%DEST%/"
if errorlevel 1 ( echo scp train failed & exit /b 1 )
if exist "%RX_ROOT%\test" scp %KEYOPT% -r "%RX_ROOT%\test" "%HOST%:%DEST%/"

echo.
echo Done. On the instance:  RX_ROOT=%DEST% NPROC=8 ./train_horde.sh run.steps=20000
endlocal
