@echo off
REM Generate video from world-model checkpoint using RacerX test data
REM
REM Usage:
REM   .\generate_video_racerx.bat                                           (default checkpoint + RacerX video)
REM   .\generate_video_racerx.bat outputs\wm_ckpt_98000.pth                (custom checkpoint + RacerX video)
REM   .\generate_video_racerx.bat outputs\wm_ckpt_98000.pth data\my_video.mp4  (custom checkpoint + custom video)

setlocal enabledelayedexpansion

REM Get checkpoint from argument or use default
set "CKPT=%~1"
if "!CKPT!"=="" (
    set "CKPT=outputs\wm_ckpt_49000.pth"
    echo [INFO] No checkpoint specified, using default: !CKPT!
)

if not exist "!CKPT!" (
    echo ERROR: Checkpoint not found: !CKPT!
    echo.
    echo Available checkpoints:
    dir /b outputs\wm_ckpt_*.pth
    exit /b 1
)

REM Get video path from second argument (optional)
set "VIDEO=%~2"

if not "!VIDEO!"=="" (
    REM Custom video path provided
    if not exist "!VIDEO!" (
        echo ERROR: Video not found: !VIDEO!
        exit /b 1
    )
    echo Using checkpoint: !CKPT!
    echo Using video: !VIDEO!
    goto :run
)

REM Try RacerX test data first
for /f "delims=" %%F in ('dir /b C:\recordings\mira_wds\test\000\*.mp4 2^>nul ^| findstr /r ".*"') do (
    set "VIDEO=C:\recordings\mira_wds\test\000\%%F"
    echo Using checkpoint: !CKPT!
    echo Using RacerX video: !VIDEO!
    goto :run
)

REM Fall back to local data/ directory
for /f "delims=" %%F in ('dir /b data\*.mp4 2^>nul ^| findstr /r ".*"') do (
    set "VIDEO=data\%%F"
    echo Using checkpoint: !CKPT!
    echo Using local video: !VIDEO!
    goto :run
)

echo ERROR: No videos found in:
echo   - C:\recordings\mira_wds\test\000\ (RacerX)
echo   - data\ (local)
echo.
echo Usage: .\generate_video_racerx.bat [checkpoint] [video_path]
echo Example: .\generate_video_racerx.bat outputs\wm_ckpt_98000.pth data\my_video.mp4
exit /b 1

:run
REM Extract checkpoint step and clip ID for output naming
for %%F in ("!CKPT!") do set "CKPTNAME=%%~nF"
set "CKPTNAME=!CKPTNAME:.pth=!"

REM Extract clip ID from video filename
for %%F in ("!VIDEO!") do set "FILENAME=%%~nF"

REM Try to extract c##### from filename
for /f "tokens=*" %%A in ("!FILENAME!") do (
    set "STR=%%A"
    for /f "delims=c" %%B in ("!STR!") do set "REST=c%%B"
    if "!REST:~0,1!"=="c" set "CLIPID=!REST:~0,6!"
)

REM If clip ID not found, use generic name
if "!CLIPID!"=="" set "CLIPID=c00000"

set "OUTPUT=outputs\generated_native_wm_ckpt_!CKPTNAME!_!CLIPID!.mp4"

echo Output: !OUTPUT!
echo.

.\.venv\Scripts\python.exe generate_from_inference.py "!CKPT!" --video "!VIDEO!" --output "!OUTPUT!"

if errorlevel 1 (
    echo ERROR: Video generation failed
    exit /b 1
)

echo.
echo Generated video: !OUTPUT!
pause
