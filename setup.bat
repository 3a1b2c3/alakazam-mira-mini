@echo off
setlocal enableextensions enabledelayedexpansion
REM ==========================================================================
REM MIRA setup: uv .venv (Python 3.11) + CUDA torch + editable install.
REM   - Python 3.11 (NOT 3.12: snapshot_download/tqdm thread pools deadlock on Windows)
REM   - torch from the cu128 index FIRST (Blackwell/RTX 5090 sm_120; never CPU wheel),
REM     so mira's deps resolve against the CUDA torch
REM   - mira editable with the hf/hydra/viz extras (loading data + configs + viz).
REM     NOTE: the `decode` extra (torchcodec) is version-touchy (wants torch 2.8.0 +
REM     FFmpeg 7); left out here -- add it later if you need video decoding.
REM Usage:  setup.bat
REM ==========================================================================
REM Build the .venv IN the sibling mira repo (editable install must run from mira's pyproject).
set "MIRA=%~dp0..\mira"
cd /d "%MIRA%"
set "VENV=%MIRA%\.venv"
set "PY=%VENV%\Scripts\python.exe"

set "PATH=%USERPROFILE%\.local\bin;%PATH%"
where uv >nul 2>nul || ( echo ERROR: uv not on PATH & exit /b 1 )
for /f "delims=" %%U in ('where uv') do set "UV_EXE=%%U" & goto :gotuv
:gotuv
REM strip ambient venv state so we build cleanly
set "VIRTUAL_ENV="
set "PYTHONHOME="
set "PYTHONPATH="

if not exist "%PY%" (
    echo --- creating venv %VENV% (Python 3.11) ---
    "!UV_EXE!" venv "%VENV%" --python 3.11 || ( echo ERROR: uv venv failed & exit /b 1 )
)

echo --- installing torch + torchvision (cu128, Blackwell) ---
"!UV_EXE!" pip install --python "%PY%" torch torchvision --index-url https://download.pytorch.org/whl/cu128 || ( echo ERROR: torch install failed & exit /b 1 )

echo --- installing mira (editable) + hf,hydra,viz extras ---
"!UV_EXE!" pip install --python "%PY%" -e ".[hf,hydra,viz]" || ( echo ERROR: mira install failed & exit /b 1 )

echo.
echo --- verify ---
"%PY%" -c "import torch, mira; print('torch', torch.__version__, 'cuda', torch.cuda.is_available()); print('mira ok')"
echo.
echo Setup complete: %VENV%
echo Next: download_models.bat  (rocket-science dataset)
endlocal
