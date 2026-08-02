@echo off
setlocal enableextensions
REM ==========================================================================
REM Pull the NEWEST checkpoint from a Horde node to a local dir. HOST is
REM PROMPTED (so RUN + DEST are the positional args -- no host/run mix-up).
REM Default = model only (checkpoint.pth, ~7.8 GB; skips the 22 GB optimizer).
REM Connects as horde@<host> with your working key (NOT the broken `horde` alias).
REM
REM Usage:
REM   pull_ckpt.bat                                    REM finetune run -> C:\recordings\mira_ckpt
REM   pull_ckpt.bat <REMOTE_RUN>                       REM e.g. mira/train_world_model_logs_finetune
REM   pull_ckpt.bat <REMOTE_RUN> <LOCAL_DEST>          REM custom dest
REM   pull_ckpt.bat <REMOTE_RUN> <LOCAL_DEST> full     REM whole 29 GB dir (incl. optimizer)
REM ==========================================================================
set "USER=horde"
set "PORT=22"
set "RUN=%~1"
if "%RUN%"=="" set "RUN=mira/train_world_model_logs_finetune"
set "DEST=%~2"
if "%DEST%"=="" set "DEST=C:\recordings\mira_ckpt"
set "MODE=%~3"

echo Nodes:  moqtul = 10.57.233.24    jzs62v = 10.57.233.223
set /p "HOST=Host [10.57.233.24]: "
if "%HOST%"=="" set "HOST=10.57.233.24"

echo.
echo Finding newest checkpoint on %USER%@%HOST%:%RUN% ...
set "CKPT="
for /f "usebackq delims=" %%C in (`ssh -p %PORT% %USER%@%HOST% "ls -d %RUN%/checkpoint-*/ 2>/dev/null | sed 's:/$::' | sort -t- -k2 -n | tail -1"`) do set "CKPT=%%C"
if "%CKPT%"=="" ( echo ERROR: no checkpoint in %RUN% on %HOST% ^(check the run dir / host^) & exit /b 1 )
echo   newest : %CKPT%
echo   dest   : %DEST%
if not exist "%DEST%" mkdir "%DEST%"
for /f "usebackq delims=" %%D in (`wsl wslpath -u "%DEST%"`) do set "WDEST=%%D"
echo.
if /i "%MODE%"=="full" (
    echo   mode: FULL ^(29 GB, incl. optimizer^)
    wsl rsync -avz --partial --progress -e "ssh -p %PORT%" "%USER%@%HOST%:%CKPT%" "%WDEST%/"
) else (
    echo   mode: model-only ^(~7.8 GB, checkpoint.pth^)
    wsl rsync -avz --partial --progress --exclude training_state.pth -e "ssh -p %PORT%" "%USER%@%HOST%:%CKPT%" "%WDEST%/"
)
echo.
echo Done -^> %DEST%
endlocal
