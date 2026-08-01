#!/usr/bin/env bash
# Linux mirror of train.bat. Train the MIRA latent world model on the FROZEN
# mira-mini codec (125k), from scratch (no run.finetune_from). The sibling mira
# repo (../mira) supplies scripts/configs and the pixi env.
#
# Env:
#   RUN     env prefix (default "pixi run --frozen"; "" if torch on PATH / venv active)
#   CODEC   frozen codec .pth  (default: globbed from the mira-mini HF snapshot)
#   TRAIN_INDEX / TEST_INDEX   data indices (default: sourced from mira/data_paths.sh
#                              written by get_data.sh; else fall back to
#                              $MIRA_WDS/{train,test}/index.json. A dataset.train_index=
#                              override in "$@" wins regardless)
#   MIRA_WDS  local WebDataset root (default /home/horde/mira_wds) used when there is
#                              no data_paths.sh and no explicit TRAIN_INDEX
#   WORKERS dataloader workers (default 4 -- Linux has no Windows worker-spawn cost)
#
# For the full 1B: add   model/latent_world_model=1b
# For 4-player:    add   model=multi_wrapper_world_model dataset.n_players=4
#   ./train.sh run.steps=200 run.batch_size=1
set -uo pipefail
export HYDRA_FULL_ERROR=1
here="$(cd "$(dirname "$0")" && pwd)"
mira="$here/../mira"
[ -f "$mira/pixi.toml" ] || { echo "ERROR: mira trainer not found at $mira (clone it beside this repo)"; exit 1; }
RUN="${RUN:-pixi run --frozen}"
WORKERS="${WORKERS:-4}"

# Frozen codec from the downloaded mira-mini bundle (globbed -> no hardcoded hash).
if [ -z "${CODEC:-}" ]; then
    for s in "$HOME"/.cache/huggingface/hub/models--alakazamworld--mira-mini/snapshots/*/; do
        [ -f "$s/codec/checkpoint-125000/checkpoint.pth" ] && CODEC="$s/codec/checkpoint-125000/checkpoint.pth"
    done
fi
[ -n "${CODEC:-}" ] && [ -f "$CODEC" ] || { echo "ERROR: codec not found (run ./download_weights.sh 1b, or set CODEC=)"; exit 1; }

# Default data indices: sourced from data_paths.sh (get_data.sh) unless already set.
[ -z "${TRAIN_INDEX:-}" ] && [ -f "$mira/data_paths.sh" ] && . "$mira/data_paths.sh"
# Fallback: the local WebDataset at $MIRA_WDS (default /home/horde/mira_wds).
# Standard layout is {split}/index.json; also accept a flat index.json.
MIRA_WDS="${MIRA_WDS:-/home/horde/mira_wds}"
if [ -z "${TRAIN_INDEX:-}" ]; then
    if [ -f "$MIRA_WDS/train/index.json" ]; then
        TRAIN_INDEX="$MIRA_WDS/train/index.json"
        [ -f "$MIRA_WDS/test/index.json" ] && TEST_INDEX="$MIRA_WDS/test/index.json"
    elif [ -f "$MIRA_WDS/index.json" ]; then
        TRAIN_INDEX="$MIRA_WDS/index.json"
    fi
fi
idx=()
if [ -n "${TRAIN_INDEX:-}" ]; then
    idx=(dataset.train_index="$TRAIN_INDEX" dataset.test_index="${TEST_INDEX:-$TRAIN_INDEX}")
fi
# NOTE: no idx and no dataset.train_index in "$@" -> the trainer uses its config default.

cd "$mira"
export WANDB_MODE=offline

# Auto-detect and resume from latest checkpoint if it exists
output_dir=""
for arg in "$@"; do
    if [[ "$arg" == run.output_dir=* ]]; then
        output_dir="${arg#run.output_dir=}"
        break
    fi
done

# Use scratch dir by default (separate from finetune)
if [ -z "$output_dir" ]; then
    output_dir="train_world_model_logs_scratch"
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

echo "GPU free:"
nvidia-smi --query-gpu=memory.free --format=csv,noheader || true
echo "codec = $CODEC"
[ -n "${TRAIN_INDEX:-}" ] && { echo "train = $TRAIN_INDEX"; echo "test  = ${TEST_INDEX:-$TRAIN_INDEX}"; }
echo

exec $RUN python scripts/train_world_model.py \
    model.architecture.config.codec_checkpoint="$CODEC" \
    "${idx[@]}" \
    run.batch_size=1 run.compile=false wandb.mode=disabled dataloader.num_workers="$WORKERS" run.log_every=50 validation.downstream_val_every=5000 \
    run.output_dir="$output_dir" \
    $continue_from \
    '++tensorboard.logdir=${run.output_dir}/tb' "$@"
