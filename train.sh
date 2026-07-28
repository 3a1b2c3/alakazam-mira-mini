#!/usr/bin/env bash
# Linux mirror of train.bat. Train the MIRA latent world model on the FROZEN
# mira-mini codec (125k), from scratch (no run.finetune_from). The sibling mira
# repo (../mira) supplies scripts/configs and the pixi env.
#
# Env:
#   RUN     env prefix (default "pixi run --frozen"; "" if torch on PATH / venv active)
#   CODEC   frozen codec .pth  (default: globbed from the mira-mini HF snapshot)
#   TRAIN_INDEX / TEST_INDEX   data indices (default: sourced from mira/data_paths.sh
#                              written by get_data.sh; a dataset.train_index= override
#                              in "$@" wins regardless)
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
idx=()
if [ -n "${TRAIN_INDEX:-}" ]; then
    idx=(dataset.train_index="$TRAIN_INDEX" dataset.test_index="${TEST_INDEX:-$TRAIN_INDEX}")
fi
# NOTE: no idx and no dataset.train_index in "$@" -> the trainer uses its config default.

cd "$mira"
export WANDB_MODE=offline
echo "GPU free:"
nvidia-smi --query-gpu=memory.free --format=csv,noheader || true
echo "codec = $CODEC"
[ -n "${TRAIN_INDEX:-}" ] && { echo "train = $TRAIN_INDEX"; echo "test  = ${TEST_INDEX:-$TRAIN_INDEX}"; }
echo

exec $RUN python scripts/train_world_model.py \
    model.architecture.config.codec_checkpoint="$CODEC" \
    "${idx[@]}" \
    run.batch_size=1 run.compile=false wandb.mode=offline dataloader.num_workers="$WORKERS" "$@"
