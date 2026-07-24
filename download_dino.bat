@echo off
setlocal enableextensions enabledelayedexpansion
REM ==========================================================================
REM Download the (Meta-gated) DINOv3 backbone weights into mira's dino_weights\,
REM used by codec TRAINING (vitl16, large) and the world-model FDD METRIC that
REM eval_wm.bat runs (vitb16, base). The weights are gated, so you must supply
REM your own time-limited SIGNED download URLs via env vars (get them from
REM https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/):
REM
REM   set DINOV3_VITL16_URL=<signed url for the large weights>
REM   set DINOV3_VITB16_URL=<signed url for the base  weights>
REM   download_dino.bat
REM
REM Skips any file already present, so you can run it again after adding a URL.
REM ==========================================================================
set "MIRA=%~dp0..\mira"
set "DEST=%MIRA%\dino_weights"
if not exist "%DEST%" mkdir "%DEST%"

call :get "%DINOV3_VITL16_URL%" "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth" "vitl16 (large, codec training)"
call :get "%DINOV3_VITB16_URL%" "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth" "vitb16 (base, eval_wm metrics)"

echo.
echo dino_weights now holds:
dir /b "%DEST%\*.pth" 2>nul
endlocal
goto :eof

:get
REM %1=signed url  %2=target filename  %3=label
set "URL=%~1"
set "FN=%~2"
if exist "%DEST%\%FN%" ( echo   have %~3: %FN% & goto :eof )
if "%URL%"=="" ( echo   SKIP %~3: set its URL env var to download %FN% & goto :eof )
echo   downloading %~3 -^> %FN%
curl -L --fail -o "%DEST%\%FN%" "%URL%"
if errorlevel 1 ( echo   ERROR: download failed for %FN% ^(check the signed URL/expiry^) )
goto :eof
