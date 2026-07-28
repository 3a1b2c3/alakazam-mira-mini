@echo off
setlocal enableextensions
REM ==========================================================================
REM Upload mira_wds\{train,test} to the specific Horde instance kschmid-jzs62v
REM via scp (Windows OpenSSH). The PASSWORD is NOT stored here -- ssh/scp prompt
REM for it and you type it (storing a password in a script logs it to disk).
REM
REM >>> VERIFY the 3 connection fields below against the Horde UI's SSH line <<<
REM     (e.g. `ssh kschmid@1.2.3.4 -p 2222` -> USER=kschmid HOST=1.2.3.4 PORT=2222)
REM
REM PREREQ: rebuild the WebDataset first if shards are missing
REM   build_mira_webdataset.py --rebuild --out C:\recordings\mira_wds\train --workers 8
REM ==========================================================================

REM --- connection (EDIT to match Horde's SSH panel) -------------------------
set "USER=horde"
set "HOST=10.57.233.223"
set "PORT=22"
REM --------------------------------------------------------------------------

REM home-relative (no leading /): lands in ~/mira_wds -- /data needs root, home is writable.
set "DEST=mira_wds"
if not defined RX_ROOT set "RX_ROOT=C:\recordings\mira_wds"
if not exist "%RX_ROOT%\train\index.json" ( echo ERROR: no data at %RX_ROOT%\train -- build the WebDataset first & exit /b 1 )

echo Target : %USER%@%HOST%  port %PORT%  -^>  %DEST%
echo Source : %RX_ROOT%\{train,test}
echo (SSH will prompt for the password -- type it; it is NOT stored)
echo.

ssh -p %PORT% %USER%@%HOST% "mkdir -p '%DEST%'"
if errorlevel 1 ( echo ssh failed -- check USER/HOST/PORT against the Horde SSH line & exit /b 1 )

scp -P %PORT% -r "%RX_ROOT%\train" "%USER%@%HOST%:%DEST%/"
if errorlevel 1 ( echo scp train failed & exit /b 1 )
if exist "%RX_ROOT%\test" scp -P %PORT% -r "%RX_ROOT%\test" "%USER%@%HOST%:%DEST%/"

echo.
echo Done. On the instance:  RX_ROOT=$HOME/mira_wds NPROC=4 ./finetune_racerx.sh
endlocal
