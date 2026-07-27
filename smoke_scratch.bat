@echo off
setlocal enableextensions
REM ==========================================================================
REM FROM-SCRATCH smoke: random-init the world model on the frozen mira-mini codec and run a tiny
REM capped pass on the RacerX data to prove the from-scratch path runs end-to-end. Validation is
REM capped (val_n_samples=8, metrics num_samples=16) so val_first doesn't stall.
REM   -> REAL from-scratch: train_racerx.bat      finetune smoke: smoke_finetune.bat
REM Uses the held-out test/ split for eval if present (else reuses train, warns).
REM
REM   smoke_scratch.bat                  200-step from-scratch smoke
REM   smoke_scratch.bat run.steps=50     even shorter
REM ==========================================================================
set "HERE=%~dp0"
set "MIRA=%~dp0..\mira"

REM Data root derived relative to this bat (%%~fI resolves ..\..\.. -- no hardcoded full paths).
for %%I in ("%~dp0..\..\..\recordings\mira_wds") do set "REC=%%~fI"
set "RECF=%REC:\=/%"
set "RX_ARGS="
if exist "%REC%\train\index.json" (
    if exist "%REC%\test\index.json" (
        set "RX_ARGS=dataset.train_index=%RECF%/train/index.json dataset.test_index=%RECF%/test/index.json"
        echo data = racer-x  train + held-out test split
    ) else (
        set "RX_ARGS=dataset.train_index=%RECF%/train/index.json dataset.test_index=%RECF%/train/index.json"
        echo data = racer-x  train  ^(test REUSES train -- run holdout_wds.py for a real split^)
    )
) else (
    echo data = rocket-science fallback  ^(no racer-x index under %REC%^)
)

echo === GPU free (need a few GB; ~0 MiB means something else is holding it) ===
nvidia-smi --query-gpu=memory.free,memory.used --format=csv,noheader
echo.

REM train.bat = from scratch (no run.finetune_from). Capped validation so val_first doesn't stall.
call "%HERE%train.bat" run.steps=200 validation.val_first=true validation.val_n_samples=8 world_model_metrics.num_samples=16 tensorboard.logdir=null %RX_ARGS% %*

echo.
echo === from-scratch smoke done. Logs/checkpoints under: %MIRA%\train_world_model_logs ===
echo If a checkpoint appeared and it reached step 200, the from-scratch harness is good.
endlocal
