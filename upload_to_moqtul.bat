@echo off
setlocal enableextensions enabledelayedexpansion
REM ==========================================================================
REM Upload mira_wds\{train,test} to the specific Horde instance kschmid-moqtul.
REM SKIP-EXISTING: uses `wsl rsync` -- shards already on the instance (matching
REM size) are skipped, and a partially-sent .tar is RESUMED, not restarted.
REM Falls back to scp -r (re-sends everything) if WSL can't reach the host.
REM The PASSWORD is NOT stored -- rsync/scp prompt for it and you type it.
REM
REM >>> VERIFY the 3 connection fields below against the Horde UI's SSH line <<<
REM ==========================================================================

REM --- connection (EDIT to match Horde's SSH panel) -------------------------
set "USER=horde"
set "HOST=10.57.233.24"
set "PORT=22"
REM --------------------------------------------------------------------------

REM home-relative (no leading /): lands in ~/mira_wds -- home is writable, /data is not.
set "DEST=mira_wds"
if not defined RX_ROOT set "RX_ROOT=C:\recordings\mira_wds"
if not exist "%RX_ROOT%\train\index.json" ( echo ERROR: no data at %RX_ROOT%\train -- build the WebDataset first & exit /b 1 )

echo Target : %USER%@%HOST%  port %PORT%  -^>  %DEST%
echo Source : %RX_ROOT%\{train,test}
echo (SSH will prompt for the password -- type it; it is NOT stored)
echo.

ssh -p %PORT% %USER%@%HOST% "mkdir -p '%DEST%'"
if errorlevel 1 ( echo ssh failed -- check USER/HOST/PORT against the Horde SSH line & exit /b 1 )

REM --- try skip-existing rsync via WSL --------------------------------------
REM wslpath translates C:\recordings\mira_wds -> /mnt/c/recordings/mira_wds cleanly.
REM Forward-slash the path BEFORE wslpath -- cmd/wsl eat backslashes in quoted args
REM (C:\a\b -> C:ab), so C:/a/b survives and wslpath -u -> /mnt/c/a/b (lowercase mount).
set "FSRC=%RX_ROOT:\=/%"
set "WSLSRC="
for /f "delims=" %%p in ('wsl wslpath -u "%FSRC%" 2^>nul') do set "WSLSRC=%%p"
if defined WSLSRC (
    echo == rsync ^(skip-existing + resume partial^) ==
    REM trailing slash on src -> copy CONTENTS (train/, test/) into DEST/. --size-only
    REM skips remote files already at full size (scp doesn't preserve mtime, so a plain
    REM mtime+size check would re-send them); --partial resumes an interrupted .tar.
    REM sync ONLY train/ + test/ (no trailing slash -> land as DEST/train, DEST/test);
    REM never the wm_* output dirs / checkpoints / wandb that also live under mira_wds.
    REM !RSRC! (delayed) -- RSRC is set INSIDE this block, so %RSRC% would expand
    REM empty at parse time and rsync would get no source (just lists the remote).
    set "RSRC=!WSLSRC!/train"
    if exist "%RX_ROOT%\test" set "RSRC=!WSLSRC!/train !WSLSRC!/test"
    wsl rsync -avz --partial --size-only --progress --exclude "wm_*" --exclude "wandb" -e "ssh -p %PORT%" !RSRC! "%USER%@%HOST%:%DEST%/"
    if not errorlevel 1 (
        echo.
        echo Done ^(rsync^). On the instance:  RX_ROOT=$HOME/mira_wds NPROC=4 ./finetune_racerx.sh
        endlocal & exit /b 0
    )
    echo rsync failed ^(WSL may not reach the host^) -- falling back to scp...
)

REM --- fallback: scp -r (re-sends everything; no skip) ----------------------
scp -P %PORT% -r "%RX_ROOT%\train" "%USER%@%HOST%:%DEST%/"
if errorlevel 1 ( echo scp train failed & exit /b 1 )
if exist "%RX_ROOT%\test" scp -P %PORT% -r "%RX_ROOT%\test" "%USER%@%HOST%:%DEST%/"

echo.
echo Done ^(scp^). On the instance:  RX_ROOT=$HOME/mira_wds NPROC=4 ./finetune_racerx.sh
endlocal
