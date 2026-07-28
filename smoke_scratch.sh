#!/usr/bin/env bash
# Linux mirror of smoke_scratch.bat. FROM-SCRATCH smoke: random-init the world
# model on the frozen mira-mini codec and run a tiny capped pass on the RacerX data
# to prove the from-scratch path runs end-to-end. Validation is capped
# (val_n_samples=8, metrics num_samples=16) so val_first doesn't stall.
#   -> REAL from-scratch: ./train_racerx.sh      finetune smoke: ./smoke_finetune.sh
# Uses the held-out test/ split for eval if present (else reuses train, warns).
#
#   RX_ROOT   mira_wds dir (holds train/ + test/); default /mnt/c/recordings/mira_wds
#   ./smoke_scratch.sh                  200-step from-scratch smoke
#   ./smoke_scratch.sh run.steps=50     even shorter
set -uo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
mira="$here/../mira"

REC="${RX_ROOT:-/mnt/c/recordings/mira_wds}"
RX_ARGS=()
if [ -f "$REC/train/index.json" ]; then
    if [ -f "$REC/test/index.json" ]; then
        RX_ARGS=(dataset.train_index="$REC/train/index.json" dataset.test_index="$REC/test/index.json")
        echo "data = racer-x  train + held-out test split"
    else
        RX_ARGS=(dataset.train_index="$REC/train/index.json" dataset.test_index="$REC/train/index.json")
        echo "data = racer-x  train  (test REUSES train -- run holdout_wds.py for a real split)"
    fi
else
    echo "data = rocket-science fallback  (no racer-x index under $REC)"
fi

echo "=== GPU free (need a few GB; ~0 MiB means something else is holding it) ==="
nvidia-smi --query-gpu=memory.free,memory.used --format=csv,noheader || true
echo

# train.sh = from scratch (no run.finetune_from). Capped validation so val_first doesn't stall.
"$here/train.sh" run.steps=200 validation.val_first=true validation.val_n_samples=8 world_model_metrics.num_samples=16 ++tensorboard.logdir=null "${RX_ARGS[@]}" "$@"

echo
echo "=== from-scratch smoke done. Logs/checkpoints under: $mira/train_world_model_logs ==="
echo "If a checkpoint appeared and it reached step 200, the from-scratch harness is good."
