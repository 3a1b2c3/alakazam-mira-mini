@echo off
setlocal enableextensions
set "HYDRA_FULL_ERROR=1"
REM ==========================================================================
REM Finetune (warm-start) from the released 1B world-model checkpoint in the
REM mira-mini bundle, instead of training from scratch -- the sane single-GPU
REM path. run.finetune_from loads the WEIGHTS only (fresh optimizer/step); the
REM default model config is already 1b, matching this checkpoint.
REM Prereq: get_data.bat. Append Hydra overrides, e.g. finetune.bat run.steps=200
REM ==========================================================================
set "HERE=%~dp0"
set "WM_CKPT=C:/Users/kschmid/.cache/huggingface/hub/models--alakazamworld--mira-mini/snapshots/19d668ac39814e394ae4a8f698690f761facf437/checkpoint-52000/checkpoint.pth"
call "%HERE%train.bat" run.finetune_from="%WM_CKPT%" %*
endlocal
