@echo off
setlocal enableextensions
REM ==========================================================================
REM Run the RacerX FINETUNE (outputs\finetune_ch bundle) in the flashdreams-mira
REM serving stack via WSL. Uses the mira-finetune-1p-8-step demo (added to
REM mira_car_soccer.yaml) which points at the local bundle.
REM
REM Modes:
REM   play_finetune.bat                         HEADLESS: generate mira.mp4 from a scripted action seq
REM   play_finetune.bat headless "W@10,W+D@5"   HEADLESS with a custom action script
REM   play_finetune.bat webrtc                  WEBRTC: browser UI to drive it live (prints a URL)
REM
REM First run installs flashdreams-mira into the WSL .venv (uv sync).
REM No HF_TOKEN needed: the LOCAL bundle skips the HF download, and the DINO weights come from the
REM checkpoint (encoder loads pretrained=False). Only the facebookresearch/dinov3 model DEFINITION
REM is pulled via torch.hub (public GitHub, cached once) -- needs internet on first run only.
REM ==========================================================================
set "FD=/mnt/c/workspace/world/flashdreams"
set "DEMO=mira-finetune-1p-8-step"
set "MODE=%~1"
if "%MODE%"=="" set "MODE=headless"
set "ACTS=%~2"
if "%ACTS%"=="" set "ACTS=W+D@5,W+A@5,Space@6,W@5"

echo === ensuring flashdreams-mira is installed (WSL, first run is slow) ===
wsl bash -lc "cd %FD% && uv sync --package flashdreams-mira --extra dev"
if errorlevel 1 ( echo ERROR: uv sync failed ^(HF_TOKEN set? uv on PATH in WSL?^) & exit /b 1 )

if /i "%MODE%"=="webrtc" (
    echo.
    echo === WEBRTC: watch below for the ^<IP^>/request_session URL, open it in a browser ===
    wsl bash -lc "cd %FD% && uv run --package flashdreams-mira mira-webrtc --manifest mira_car_soccer.yaml --demo %DEMO% --host 0.0.0.0 --port 8083"
) else (
    echo.
    echo === HEADLESS generate ^(demo=%DEMO%  actions=%ACTS%^) ===
    wsl bash -lc "cd %FD% && LOGURU_LEVEL=WARNING uv run --package flashdreams-mira flashdreams-run mira --manifest mira_car_soccer.yaml --demo %DEMO% --action-script '%ACTS%'"
    echo.
    echo Output MP4 + metrics under: %FD%/artifacts/mira/^<demo^>/
)
endlocal
