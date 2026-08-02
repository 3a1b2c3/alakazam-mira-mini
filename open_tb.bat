@echo off
setlocal enableextensions
REM ==========================================================================
REM Open the HORDE box's TensorBoard in your LOCAL Windows browser via an SSH
REM tunnel. Starts tensorboard ON horde (over ssh) pointed at the mira logs and
REM forwards its port to localhost, then opens the page here.
REM
REM The mira logs on horde include train_world_model_logs*/tb (loss curves) AND
REM train_world_model_logs_scratch/eval_tb (the eval/psnr the checkpoint watcher
REM writes) -- pointing at the parent /home/horde/mira shows them all as runs.
REM
REM Usage:
REM   open_tb.bat <ssh-host>            e.g.  open_tb.bat horde@kschmid-moqtul
REM   open_tb.bat <ssh-host> <logdir>  custom logdir (default /home/horde/mira)
REM   open_tb.bat <ssh-host> <logdir> <port>
REM (ssh-host is whatever you ssh into -- the prompt shows horde@<box>.)
REM Ctrl+C in this window stops both the tunnel and the remote tensorboard.
REM ==========================================================================
set "HOST=%~1"
if "%HOST%"=="" (
    echo Usage: open_tb.bat ^<ssh-host^> [logdir] [port]
    echo   e.g. open_tb.bat horde@kschmid-moqtul
    exit /b 1
)
set "LOGDIR=%~2"
if "%LOGDIR%"=="" set "LOGDIR=/home/horde/mira"
set "PORT=%~3"
if "%PORT%"=="" set "PORT=6006"

echo Tunneling %HOST% port %PORT% to localhost:%PORT% - starting TensorBoard on horde
echo   logdir: %LOGDIR%   ^| Ctrl+C to stop
start "" "http://localhost:%PORT%/"

REM -L forwards local:PORT to the remote's localhost:PORT; the remote command
REM starts tensorboard bound to localhost there (tensorboard.sh auto-installs it
REM if missing). If a TB is ALREADY on that port, tensorboard.sh exits with a bind
REM error -- fall back to 'sleep infinity' so the ssh session (and the tunnel)
REM stays open and forwards to the TB already running.
REM Free the port by NUMBER (fuser), not by name -- 'pkill -f tensorboard' would
REM also match THIS ssh command's own line (it contains 'tensorboard.sh') and kill
REM itself, dropping the tunnel instantly.
ssh -L %PORT%:localhost:%PORT% %HOST% "fuser -k %PORT%/tcp >/dev/null 2>&1; sleep 1; cd ~/alakazam-mira-mini && (bash tensorboard.sh %LOGDIR% %PORT% || (echo 'tensorboard failed to start - does the logdir have tfevents? holding tunnel, Ctrl+C to stop' && sleep infinity))"
endlocal
