@echo off
setlocal enableextensions
REM ==========================================================================
REM View mira training in TensorBoard locally. Reads the scalar logs the trainer
REM writes (++tensorboard.logdir=${run.output_dir}/tb, ON by default). Defaults to
REM the parent mira dir so EVERY run (train_world_model_logs*/tb: scratch,
REM finetune, ...) shows up as a separate, comparable TensorBoard run.
REM Uses this repo's .venv tensorboard.exe (install: uv pip install --python
REM .venv\Scripts\python.exe tensorboard).
REM Usage:
REM   tensorboard.bat                  -> serve C:\workspace\world\mira on :6006
REM   tensorboard.bat <logdir>         custom logdir
REM   tensorboard.bat <logdir> <port>  custom logdir + port
REM
REM NOTE: the LIVE run is on the horde box -> /home/horde/mira/train_world_model_logs_scratch/tb
REM (NOT local). To view it here: run tensorboard.sh ON horde + tunnel
REM   ssh -L 6006:localhost:6006 <horde-host>   (or copy the tb dir down and pass it).
REM ==========================================================================
set "TB=%~dp0.venv\Scripts\tensorboard.exe"
if not exist "%TB%" set "TB=%~dp0..\mira\.venv\Scripts\tensorboard.exe"
if not exist "%TB%" (
    echo ERROR: tensorboard.exe not found. Install once:
    echo   uv pip install --python "%~dp0.venv\Scripts\python.exe" tensorboard
    exit /b 1
)

set "LOGDIR=%~1"
if "%LOGDIR%"=="" set "LOGDIR=C:\workspace\world\mira"
set "PORT=%~2"
if "%PORT%"=="" set "PORT=6006"

if not exist "%LOGDIR%" ( echo ERROR: logdir not found: %LOGDIR%  ^(has training written any TB events yet?^) & exit /b 1 )

echo Serving TensorBoard for "%LOGDIR%" on http://localhost:%PORT%/  ^(Ctrl+C to stop^)
start "" "http://localhost:%PORT%/"
"%TB%" --logdir "%LOGDIR%" --port %PORT% --host localhost
endlocal
