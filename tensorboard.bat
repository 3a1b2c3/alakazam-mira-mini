@echo off
setlocal enableextensions
REM ==========================================================================
REM Launch TensorBoard for MIRA training runs. Reads the scalar logs the
REM trainer writes (tensorboard.logdir=${run.output_dir}/tb, ON by default).
REM Points at the parent C:\recordings\mira_wds so BOTH codec_smoke\tb and
REM wm_smoke\tb (and any other run dir) show up as separate runs.
REM Uses mira's .venv tensorboard.exe (python -m tensorboard has no __main__).
REM Usage:
REM   tensorboard.bat                 -> serve C:\recordings\mira_wds on :6006
REM   tensorboard.bat <logdir>        custom logdir
REM   tensorboard.bat <logdir> <port> custom logdir + port
REM ==========================================================================
set "TB=C:\workspace\world\mira\.venv\Scripts\tensorboard.exe"
if not exist "%TB%" ( echo ERROR: tensorboard.exe not found at %TB% - is mira's .venv set up? & exit /b 1 )

set "LOGDIR=%~1"
if "%LOGDIR%"=="" set "LOGDIR=C:\recordings\mira_wds"
set "PORT=%~2"
if "%PORT%"=="" set "PORT=6006"

if not exist "%LOGDIR%" ( echo ERROR: logdir not found: %LOGDIR%  ^(has training written any TB events yet?^) & exit /b 1 )

echo Serving TensorBoard for "%LOGDIR%" on http://localhost:%PORT%  ^(Ctrl+C to stop^)
"%TB%" --logdir "%LOGDIR%" --port %PORT%
endlocal
