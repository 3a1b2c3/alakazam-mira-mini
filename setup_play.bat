@echo off
setlocal enableextensions enabledelayedexpansion
REM ==========================================================================
REM MIRA Mini PLAY setup: WorldCanvas .venv [Python 3.11] + CUDA torch + editable
REM install of alakazam-mira-mini, which provides the mira-mini command that
REM run_mira.bat and download_weights.bat expect.
REM   - SEPARATE from setup.bat, which builds the mira TRAINER venv.
REM   - Python 3.11 [NOT 3.12: snapshot_download/tqdm thread pools deadlock].
REM   - torch from the cu128 index FIRST [Blackwell sm_120; never CPU wheel],
REM     so alakazam-mira-mini's torch>=2.6 dep resolves against the CUDA wheel.
REM Usage:  setup_play.bat
REM ==========================================================================
set "PKG=%~dp0"
set "VENV=%~dp0.venv"
set "PY=%VENV%\Scripts\python.exe"

set "PATH=%USERPROFILE%\.local\bin;%PATH%"
where uv >nul 2>nul || ( echo ERROR: uv not on PATH & exit /b 1 )
for /f "delims=" %%U in ('where uv') do set "UV_EXE=%%U" & goto :gotuv
:gotuv
set "VIRTUAL_ENV="
set "PYTHONHOME="
set "PYTHONPATH="

if exist "%PY%" goto :havevenv
echo --- creating venv [Python 3.11] ---
"!UV_EXE!" venv "%VENV%" --python 3.11 || ( echo ERROR: uv venv failed & exit /b 1 )
:havevenv

echo --- installing torch + torchvision [cu128, Blackwell] ---
"!UV_EXE!" pip install --python "%PY%" torch torchvision --index-url https://download.pytorch.org/whl/cu128 || ( echo ERROR: torch install failed & exit /b 1 )

echo --- installing alakazam-mira-mini [editable, provides mira-mini] ---
"!UV_EXE!" pip install --python "%PY%" -e "%PKG%." || ( echo ERROR: alakazam-mira-mini install failed & exit /b 1 )

echo.
if not exist "%VENV%\Scripts\mira-mini.exe" goto :warn
echo OK: mira-mini installed at %VENV%\Scripts\mira-mini.exe
echo Now run run_mira.bat   [or download_weights.bat first]
goto :end
:warn
echo WARNING: mira-mini.exe not found after install -- check the pip output above
:end
endlocal
