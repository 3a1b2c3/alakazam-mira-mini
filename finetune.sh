#!/usr/bin/env bash
# Linux mirror of finetune.bat. Finetune (warm-start) from the released 1B
# world-model checkpoint (checkpoint-52000) in the mira-mini bundle instead of
# training from scratch -- the sane single-GPU path. run.finetune_from loads the
# WEIGHTS only (fresh optimizer/step); the default model config is already 1b.
# Delegates to train.sh (which supplies the frozen codec + data + env), so it
# inherits train.sh's data default: $MIRA_WDS/{train,test}/index.json
# (MIRA_WDS=/home/horde/mira_wds) when there's no data_paths.sh / TRAIN_INDEX.
# So ./finetune.sh alone works once mira_wds is populated; override with
# ./get_data.sh, TRAIN_INDEX=, or dataset.train_index=. Extra Hydra overrides
# pass through, e.g.  ./finetune.sh run.steps=200
set -uo pipefail
export HYDRA_FULL_ERROR=1
here="$(cd "$(dirname "$0")" && pwd)"

# Warm-start WM from the mira-mini HF snapshot (globbed -> no hardcoded hash).
WM="${WM:-}"
if [ -z "$WM" ]; then
    for s in "$HOME"/.cache/huggingface/hub/models--alakazamworld--mira-mini/snapshots/*/; do
        [ -f "$s/checkpoint-52000/checkpoint.pth" ] && WM="$s/checkpoint-52000/checkpoint.pth"
    done
fi
[ -n "$WM" ] && [ -f "$WM" ] || { echo "ERROR: warm-start checkpoint-52000 not found (run ./download_weights.sh 1b, or set WM=)"; exit 1; }

exec "$here/train.sh" run.finetune_from="$WM" "$@"
