@echo off
setlocal enableextensions
set "HYDRA_FULL_ERROR=1"
REM ==========================================================================
REM FINETUNE smoke: warm-start from the mira-mini 1B checkpoint (checkpoint-52000) and run a tiny
REM capped pass on the RacerX data to prove the pipeline runs end-to-end before a real run.
REM Percent cadences auto-scale, so 200 steps still checkpoints + validates. Validation is capped
REM (val_n_samples=8, metrics num_samples=16) so val_first doesn't stall.
REM   -> REAL finetune:  finetune_racerx.bat        from-scratch smoke: smoke_scratch.bat
REM Uses the held-out test/ split for eval if present (else reuses train, warns).
REM
REM   smoke_finetune.bat                 200-step warm-start smoke
REM   smoke_finetune.bat run.steps=50    even shorter
REM ==========================================================================
set "HERE=%~dp0"
set "MIRA=%~dp0..\mira"

REM Data root derived relative to this bat (%%~fI resolves ..\..\.. to a clean path -- no
REM hardcoded full paths). RECF = forward-slash form for Hydra. Use the held-out test/ split for
REM eval if it exists (holdout_wds.py); else fall back to reusing train (leaky, warns).
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

call "%HERE%finetune.bat" run.steps=200 validation.val_first=true validation.val_n_samples=8 world_model_metrics.num_samples=16 ++tensorboard.logdir=null %RX_ARGS% %*

echo.
echo === smoke test done. Logs/checkpoints under: %MIRA%\train_world_model_logs ===
echo If a checkpoint + a decreasing loss appeared and it reached step 200, the harness is good.
endlocal
