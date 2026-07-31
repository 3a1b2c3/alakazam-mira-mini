#!/usr/bin/env bash
# Launch RacerX world-model training on LINUX (WSL or cluster). Wires the RacerX WebDataset
# (train + held-out test split) + the mira-mini frozen codec + optional warm-start, then calls
# the mira trainer (../mira/train.sh). Run from alakazam-mira-mini; the mira repo is the sibling.
#
# Env:
#   RX_ROOT   dir holding train/ and test/ WebDatasets   (required)
#               WSL:      /mnt/c/recordings/mira_wds
#               cluster:  wherever you rsync'd it
#   CODEC     .../mira-mini/.../codec/checkpoint-125000/checkpoint.pth   (required, frozen codec)
#   WM        .../mira-mini/.../checkpoint-52000/checkpoint.pth          (optional -> warm-start/finetune)
#   RUN       env prefix (default "pixi run --frozen"; "" if torch is already on PATH / venv active)
#
# Extra Hydra overrides pass through, e.g.:
#   RX_ROOT=/mnt/c/recordings/mira_wds CODEC=<codec.pth> ./train_racerx.sh run.steps=20000
#   WM=<wm.pth> RX_ROOT=... CODEC=... ./train_racerx.sh run.steps=5000     # finetune
set -uo pipefail
export HYDRA_FULL_ERROR=1
here="$(cd "$(dirname "$0")" && pwd)"
mira="$here/../mira"
# -f not -x: git clones of the mira repo often drop the exec bit; we invoke it via `bash` below.
[ -f "$mira/train.sh" ] || { echo "ERROR: mira trainer not found at $mira (clone it beside this repo)"; exit 1; }

: "${RX_ROOT:?set RX_ROOT=/path/to/mira_wds (holds train/ and test/)}"
export DATA_INDEX="$RX_ROOT/train/index.json"
[ -f "$DATA_INDEX" ] || { echo "ERROR: no train index at $DATA_INDEX -- build the WebDataset first"; exit 1; }
if [ -f "$RX_ROOT/test/index.json" ]; then
    export TEST_INDEX="$RX_ROOT/test/index.json"          # honest eval on the held-out split
    echo "data: train + held-out test ($RX_ROOT)"
else
    echo "data: train only ($RX_ROOT) -- no test split (eval reuses train; run holdout_wds.py for one)"
fi
: "${CODEC:?set CODEC=/path/to/mira-mini/codec/checkpoint-125000/checkpoint.pth}"
# WM is optional; train.sh warm-starts if it's set.

# Auto-detect and resume from latest checkpoint if it exists
output_dir=""
for arg in "$@"; do
    if [[ "$arg" == run.output_dir=* ]]; then
        output_dir="${arg#run.output_dir=}"
        break
    fi
done

if [ -z "$output_dir" ]; then
    output_dir=$(ls -dt train_world_model_logs* 2>/dev/null | head -1)
fi

continue_from=""
if [ -n "$output_dir" ] && [ -d "$output_dir/checkpoints" ]; then
    latest_ckpt=$(find "$output_dir/checkpoints" -name "checkpoint-*.pth" 2>/dev/null | \
        sed 's/.*checkpoint-\([0-9]*\).*/\1/' | sort -n | tail -1)
    if [ -n "$latest_ckpt" ]; then
        ckpt_path="$output_dir/checkpoints/checkpoint-${latest_ckpt}.pth"
        [ -f "$ckpt_path" ] && continue_from="run.continue_from=$ckpt_path"
        if [ -n "$continue_from" ]; then
            echo "Auto-resuming from: $ckpt_path (step $latest_ckpt)"
        fi
    fi
fi

# dataloader.num_workers>0 is fine on Linux (unlike Windows); raise it for real runs via the args.
# Same baseline defaults as finetune_racerx.sh (compile=30x faster, freq checkpoints, capped
# val, TB events) -- passed through train.sh to the trainer; override any via extra args.
exec bash "$mira/train.sh" \
    run.compile=true run.checkpoint_every=250 run.log_every=50 optim.scheduler.warmup_steps=200 \
    validation.val_n_samples=64 world_model_metrics.num_samples=128 \
    $continue_from \
    '++tensorboard.logdir=${run.output_dir}/tb' "$@"
