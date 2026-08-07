@echo off
REM Generate video using real actions from JSONL file

setlocal enabledelayedexpansion

REM Find first RacerX video with action data
for /f "delims=" %%F in ('dir /b "C:\recordings\mira_wds\test\000\dataset_00000\*.mp4" 2^>nul ^| findstr /r ".*" ^| head -1') do (
    set "VIDEO=C:\recordings\mira_wds\test\000\dataset_00000\%%F"
    goto :found
)

echo ERROR: No RacerX videos found in C:\recordings\mira_wds\test\000\dataset_00000\
exit /b 1

:found
echo.
echo ============================================================
echo Generate Video with Real Actions
echo ============================================================
echo.
echo Video: !VIDEO!

REM Extract base name for output
for %%F in (!VIDEO!) do set "BASENAME=%%~nF"
set "BASENAME=!BASENAME:.p0.mp4=!"
set "OUTPUT=%CD%\outputs\generated_!BASENAME!_with_actions.mp4"

echo Output: !OUTPUT!
echo.

REM Run generation with real actions (auto-loaded from .jsonl) and longer clip
echo Running generation (loading actions from .jsonl automatically)...
.\.venv\Scripts\python.exe generate_from_inference.py outputs/wm_ckpt_49000.pth --video "!VIDEO!" --context-frames 8 --gen-frames 32 --output "!OUTPUT!"

if errorlevel 1 (
    echo ERROR: Video generation failed
    exit /b 1
)

echo.
echo ============================================================
echo Video generated. Now evaluating...
echo ============================================================
echo.

REM Run evaluation
.\.venv\Scripts\python.exe eval_video_quality.py --real "!VIDEO!" --gen "!OUTPUT!" --tb-dir outputs/tb --step 49000

if errorlevel 1 (
    echo ERROR: Evaluation failed
    exit /b 1
)

echo.
echo ============================================================
echo Done! Video and metrics saved.
echo ============================================================
echo Video:   !OUTPUT!
echo Metrics: %CD%\outputs\tb
echo.

pause
