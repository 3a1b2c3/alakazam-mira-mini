@echo off
setlocal enableextensions
REM Download 1 shard each of the rocket-science train/test splits and write
REM data_paths.bat (sourced by train.bat). Edit SHARDS in get_data.py for more.
set "MIRA=%~dp0..\mira"
"%MIRA%\.venv\Scripts\python.exe" "%MIRA%\get_data.py"
endlocal
