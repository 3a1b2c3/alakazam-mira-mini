#!/usr/bin/env bash
# Resume MIRA world-model training from the latest checkpoint.
# Auto-finds the highest-step checkpoint and continues training.
#
# Usage:
#   ./resume.sh [extra args]
#
# To specify a custom output dir:
#   ./resume.sh run.output_dir=/custom/path [extra args]
set -uo pipefail
export HYDRA_FULL_ERROR=1

here="$(cd "$(dirname "$0")" && pwd)"
mira="$here/../mira"
[ -f "$mira/train.sh" ] || { echo "ERROR: mira trainer not found at $mira"; exit 1; }

# Find latest output_dir if not explicitly set in args
output_dir=""
for arg in "$@"; do
    if [[ "$arg" == run.output_dir=* ]]; then
        output_dir="${arg#run.output_dir=}"
        break
    fi
done

if [ -z "$output_dir" ]; then
    # Auto-find latest train_world_model_logs* dir
    output_dir=$(ls -dt train_world_model_logs* 2>/dev/null | head -1)
    if [ -z "$output_dir" ]; then
        echo "ERROR: no previous run found (no train_world_model_logs* dirs)"
        echo "Start a new run with: ./train.sh"
        exit 1
    fi
fi

# Find latest checkpoint in output_dir
latest_ckpt=$(find "$output_dir/checkpoints" -name "checkpoint-*.pth" 2>/dev/null | \
    sed 's/.*checkpoint-\([0-9]*\).*/\1/' | sort -n | tail -1)

if [ -z "$latest_ckpt" ]; then
    echo "ERROR: no checkpoints found in $output_dir/checkpoints"
    exit 1
fi

ckpt_path="$output_dir/checkpoints/checkpoint-${latest_ckpt}.pth"
[ -f "$ckpt_path" ] || { echo "ERROR: checkpoint not found at $ckpt_path"; exit 1; }

echo "Resuming from: $ckpt_path (step $latest_ckpt)"
echo "Output dir:    $output_dir"
echo

exec bash "$mira/train.sh" \
    "run.continue_from=$ckpt_path" \
    "run.output_dir=$output_dir" \
    '++tensorboard.logdir=${run.output_dir}/tb' "$@"
