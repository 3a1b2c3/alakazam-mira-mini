@echo off
setlocal enableextensions enabledelayedexpansion
REM ==========================================================================
REM Download a world-model checkpoint from a Horde node to a local dir, so you
REM can eval it on the local GPU (no contention with training). Prompts for the
REM host, the run dir, and which checkpoint (lists what's on the node). rsync
REM --partial -> resumable (a checkpoint is ~29 GB). Key auth (like the uploads).
REM
REM Everything can also be passed as args to skip the prompts:
REM   sync_checkpoint.bat [HOST] [REMOTE_RUN] [REMOTE_CKPT] [LOCAL_DEST]
REM Known hosts:  moqtul=10.57.233.24   jzs62v=10.57.233.223
REM ==========================================================================
set "USER=horde"
set "PORT=22"

REM --- host ---
set "HOST=%~1"
if "%HOST%"=="" (
    echo Known hosts:  moqtul = 10.57.233.24    jzs62v = 10.57.233.223
    set /p "HOST=Host [10.57.233.24]: "
)
if "%HOST%"=="" set "HOST=10.57.233.24"

REM --- remote run dir ---
set "RUN=%~2"
if "%RUN%"=="" set /p "RUN=Remote run dir [mira/train_world_model_logs_finetune]: "
if "%RUN%"=="" set "RUN=mira/train_world_model_logs_finetune"

REM --- list checkpoints on the node, then pick ---
echo.
echo Checkpoints on %USER%@%HOST%:%RUN%
ssh -p %PORT% %USER%@%HOST% "ls -d %RUN%/checkpoint-*/ 2>/dev/null | sed 's:/$::' || echo '  (none found)'"
echo.
set "CKPT=%~3"
if "%CKPT%"=="" set /p "CKPT=Full remote checkpoint path (blank = newest): "
if "%CKPT%"=="" (
    for /f "usebackq delims=" %%C in (`ssh -p %PORT% %USER%@%HOST% "ls -d %RUN%/checkpoint-*/ 2>/dev/null | sed 's:/$::' | sort -t- -k2 -n | tail -1"`) do set "CKPT=%%C"
)
if "!CKPT!"=="" ( echo ERROR: no checkpoint selected/found & exit /b 1 )

REM --- local dest ---
set "DEST=%~4"
if "%DEST%"=="" set /p "DEST=Local dest [C:\recordings\mira_ckpt]: "
if "%DEST%"=="" set "DEST=C:\recordings\mira_ckpt"
if not exist "%DEST%" mkdir "%DEST%"

echo.
echo   host : %USER%@%HOST%
echo   ckpt : !CKPT!
echo   dest : %DEST%
echo.

where wsl >nul 2>nul || goto :scp
for /f "usebackq delims=" %%D in (`wsl wslpath -u "%DEST%"`) do set "WDEST=%%D"
REM no trailing slash on source -> lands as <DEST>\checkpoint-<step>\
wsl rsync -avz --partial --progress -e "ssh -p %PORT%" "%USER%@%HOST%:!CKPT!" "!WDEST!/"
goto :done

:scp
scp -P %PORT% -r "%USER%@%HOST%:!CKPT!" "%DEST%\"

:done
echo.
echo Synced -^> %DEST%
endlocal
