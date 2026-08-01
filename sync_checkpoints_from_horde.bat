@echo off
REM Download most recent MIRA checkpoint from Horde to Windows.
REM Setup: Ensure rsync or scp available (git bash, WSL, or standalone ssh tools).
setlocal
set "HORDE_HOST=horde"
set "HORDE_PATH=~/mira_wds/wm_racerx_ft"
set "LOCAL_PATH=C:\workspace\world\mira-checkpoints"

if not exist "%LOCAL_PATH%" mkdir "%LOCAL_PATH%"

echo Fetching latest checkpoint from Horde...
for /f "delims=" %%A in ('ssh %HORDE_HOST% "ls -d %HORDE_PATH%/checkpoint-* 2^>/dev/null ^| sed 's/.*checkpoint-//' ^| sort -n ^| tail -1"') do set "CKPT=%%A"

if "%CKPT%"=="" (
    echo ERROR: No checkpoints found
    exit /b 1
)

set "REMOTE=%HORDE_HOST%:%HORDE_PATH%/checkpoint-%CKPT%"
set "LOCAL=%LOCAL_PATH%\checkpoint-%CKPT%"

echo Downloading checkpoint-%CKPT%...
rsync -avz --progress "%REMOTE%/" "%LOCAL%/" 2>nul || scp -r "%REMOTE%/checkpoint.pth" "%LOCAL%\"

if %errorlevel% equ 0 (
    echo.
    echo ✓ Complete: %LOCAL%
) else (
    echo ✗ Failed. Install rsync/scp (git bash, WSL).
    exit /b 1
)
endlocal
