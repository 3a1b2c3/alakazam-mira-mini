@echo off
setlocal enableextensions
REM ==========================================================================
REM Copy the gated DINOv3 pretrained weights to the moqtul Horde instance so the
REM codec (PHASE-1 training) can load them via RS_DINO_WEIGHTS_DIR. Finds vitl16
REM (ViT-L/16, 300M -- required for the codec) locally in the torch hub cache or
REM ..\mira\dino_weights; also copies vitb16 (eval_wm metrics) if present.
REM No re-download needed -- reuses the file you already fetched with the signed URL.
REM The PASSWORD is NOT stored -- scp prompts for it.
REM
REM >>> SET HOST below to moqtul's IP (from the Horde SSH line) before running <<<
REM ==========================================================================

set "USER=horde"
set "HOST=10.57.233.24"
set "PORT=22"
set "DEST=dino_weights"

REM --- locate vitl16 locally (torch hub cache, then ..\mira\dino_weights) -------
set "VITL="
set "HUB=%USERPROFILE%\.cache\torch\hub\checkpoints\dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
set "MIRAD=%~dp0..\mira\dino_weights\dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
if exist "%HUB%" set "VITL=%HUB%"
if not defined VITL if exist "%MIRAD%" set "VITL=%MIRAD%"
if not defined VITL ( echo ERROR: vitl16 weights not found locally -- run download_dino.sh with a signed URL first & exit /b 1 )

REM --- optional vitb16 (eval metrics) ------------------------------------------
set "VITB="
if exist "%USERPROFILE%\.cache\torch\hub\checkpoints\dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth" set "VITB=%USERPROFILE%\.cache\torch\hub\checkpoints\dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"

echo Target : %USER%@%HOST%  port %PORT%  -^>  ~/%DEST%
echo vitl16 : %VITL%
if defined VITB echo vitb16 : %VITB%
echo (SSH prompts for the password -- not stored)
echo.

if /I "%HOST%"=="PASTE_MOQTUL_IP" ( echo ERROR: set HOST to moqtul's IP first ^(edit line 14^) & exit /b 1 )

ssh -p %PORT% %USER%@%HOST% "mkdir -p '%DEST%'" || ( echo ssh failed -- check HOST/PORT against the Horde SSH line & exit /b 1 )
scp -P %PORT% "%VITL%" "%USER%@%HOST%:%DEST%/" || ( echo scp vitl16 failed & exit /b 1 )
if defined VITB scp -P %PORT% "%VITB%" "%USER%@%HOST%:%DEST%/"

echo.
echo Done. On the instance, before phase-1 codec training:
echo   export RS_DINO_WEIGHTS_DIR=$HOME/%DEST%
endlocal
