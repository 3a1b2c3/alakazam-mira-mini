@echo off
setlocal enableextensions
REM ==========================================================================
REM Smoke-test the MIRA world-model training harness on the RACER-X data by
REM default: a tiny warm-start run (200 steps, validate at step 0) to prove the
REM pipeline runs end-to-end on a single 5090 before a full run. Percent-based
REM cadences auto-scale to run.steps, so 200 steps still checkpoints/validates.
REM
REM Data default = racer-x WebDataset (tools\replay-processing\make_mira output).
REM Only a train split exists, so test reuses train for the smoke validation.
REM Falls back to the rocket-science paths in data_paths.bat if that index is gone.
REM
REM   smoke_test.bat                    200-step run on racer-x data
REM   smoke_test.bat run.steps=50       even shorter
REM   override data: append  dataset.train_index=C:/.../index.json dataset.test_index=C:/.../index.json
REM ==========================================================================
set "HERE=%~dp0"
set "MIRA=%~dp0..\mira"

REM racer-x mira WebDataset index (forward slashes for Hydra). No test split yet.
set "RX_INDEX=C:/recordings/mira_wds/train/index.json"
set "RX_ARGS="
if exist "C:\recordings\mira_wds\train\index.json" (
    set "RX_ARGS=dataset.train_index=%RX_INDEX% dataset.test_index=%RX_INDEX%"
    echo data = racer-x  %RX_INDEX%  ^(test reuses train^)
) else (
    echo data = rocket-science fallback  ^(racer-x index not at C:\recordings\mira_wds\train\index.json^)
)

echo === GPU free (need a few GB; ~0 MiB means something else is holding it) ===
nvidia-smi --query-gpu=memory.free,memory.used --format=csv,noheader
echo.

call "%HERE%finetune.bat" run.steps=200 validation.val_first=true %RX_ARGS% %*

echo.
echo === smoke test done. Logs/checkpoints under: %MIRA%\train_world_model_logs ===
echo If a checkpoint + a decreasing loss appeared and it reached step 200, the harness is good.
endlocal
