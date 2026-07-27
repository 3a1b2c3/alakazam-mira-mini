@echo off
setlocal enableextensions
REM ==========================================================================
REM Push the racer-x mira WebDataset (C:\recordings\mira_wds\{train,test}) to a
REM PRIVATE HuggingFace dataset so Horde instances can pull it (train_horde.sh
REM HF_DATA_REPO=...). Re-run as shards grow -- upload_folder is incremental
REM (only changed/new files transfer). Uploads ONLY the shards, not the wm_* /
REM tb / wandb training outputs.
REM
REM Needs an HF token with write access: `huggingface-cli login` or set HF_TOKEN.
REM
REM   set HF_DATA_REPO=youruser/racerx-mira-wds  &  upload_racerx_data.bat
REM   set RX_ROOT=C:\recordings\mira_wds   (default)
REM ==========================================================================
if "%HF_DATA_REPO%"=="" ( echo ERROR: set HF_DATA_REPO=youruser/racerx-mira-wds & exit /b 1 )
if not defined RX_ROOT set "RX_ROOT=C:\recordings\mira_wds"
if not exist "%RX_ROOT%\train\index.json" ( echo ERROR: no data at %RX_ROOT%\train -- build the WebDataset first & exit /b 1 )

set "PY=C:\workspace\world\mira\.venv\Scripts\python.exe"
if not exist "%PY%" set "PY=python"

echo Uploading %RX_ROOT%\{train,test} -> HF dataset %HF_DATA_REPO% (private) ...
"%PY%" -c "from huggingface_hub import HfApi; import os; api=HfApi(); api.create_repo('%HF_DATA_REPO%',repo_type='dataset',private=True,exist_ok=True); [ (api.upload_folder(folder_path=os.path.join(r'%RX_ROOT%',s),repo_id='%HF_DATA_REPO%',repo_type='dataset',path_in_repo=s), print('uploaded',s)) for s in ('train','test') if os.path.isdir(os.path.join(r'%RX_ROOT%',s)) ]"
if errorlevel 1 ( echo UPLOAD FAILED & exit /b 1 )

echo.
echo Done. On the Horde instance:  HF_DATA_REPO=%HF_DATA_REPO% ./train_horde.sh
endlocal
