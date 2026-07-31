@echo off
setlocal enableextensions
REM ==========================================================================
REM DATA-ONLY sync to horde@10.57.233.223 -- rsync just train/ + test/ shards
REM (skip-existing by size, resume partial .tar), never the wm_* output dirs /
REM checkpoints / wandb. Re-run any time to top up the upload; completed shards
REM are skipped. Prompts once for the SSH password (not stored).
REM
REM   sync_data.bat                     sync C:\recordings\mira_wds\{train,test}
REM   set RX_ROOT=D:\path & sync_data.bat   custom source
REM ==========================================================================

set "USER=horde"
set "HOST=10.57.233.223"
set "PORT=22"
set "DEST=mira_wds"

if not defined RX_ROOT set "RX_ROOT=C:\recordings\mira_wds"
if not exist "%RX_ROOT%\train\index.json" ( echo ERROR: no data at %RX_ROOT%\train -- build the WebDataset first & exit /b 1 )

echo Sync   : %RX_ROOT%\{train,test}  -^>  %USER%@%HOST%:%DEST%/
echo (rsync: skip-existing + resume; SSH prompts for the password)
echo.

REM forward-slash the path before wslpath (backslashes get eaten in quoted args)
set "FSRC=%RX_ROOT:\=/%"
set "WSLSRC="
for /f "delims=" %%p in ('wsl wslpath -u "%FSRC%" 2^>nul') do set "WSLSRC=%%p"
if not defined WSLSRC ( echo ERROR: WSL not available -- use upload_to_jzs62v.bat ^(scp fallback^) & exit /b 1 )

ssh -p %PORT% %USER%@%HOST% "mkdir -p '%DEST%'" || ( echo ssh failed & exit /b 1 )

REM train + test only (no trailing slash -> land as DEST/train, DEST/test)
set "RSRC=%WSLSRC%/train"
if exist "%RX_ROOT%\test" set "RSRC=%WSLSRC%/train %WSLSRC%/test"
wsl rsync -avz --partial --size-only --progress --exclude "wm_*" --exclude "wandb" -e "ssh -p %PORT%" %RSRC% "%USER%@%HOST%:%DEST%/"

echo.
echo Done. Verify on the instance:  ls ~/%DEST%/train/*/dataset_*.tar ^| wc -l
endlocal
