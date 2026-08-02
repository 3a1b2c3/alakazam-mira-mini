@echo off
REM Plot keyboard actions inferred from physics response.
REM Shows correlation between player input and vehicle dynamics.
REM Usage: plot_actions_physics.bat [--sample 0] [--output out.png]
setlocal
python "%~dp0plot_actions_physics.py" --data C:\recordings\mira_wds\train\index.json %*
endlocal
