@echo off
REM Monitor MIRA finetune training progress on Horde.
REM Usage: monitor_training.bat [follow]
REM   (default: show last 50 lines; pass 'follow' to stream live updates)
setlocal
set "HORDE_HOST=horde"
set "LOG_PATH=~/alakazam-mira-mini/ft.log"

if /i "%~1"=="follow" (
    echo Streaming live training log from Horde...
    echo (Ctrl+C to stop)
    echo.
    ssh %HORDE_HOST% "tail -f %LOG_PATH%"
) else (
    echo Latest 50 lines of training log:
    echo.
    ssh %HORDE_HOST% "tail -50 %LOG_PATH%"
    echo.
    echo Use 'monitor_training.bat follow' to stream live updates.
)
endlocal
