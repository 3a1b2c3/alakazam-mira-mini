#!/usr/bin/env bash
# Linux mirror of train.bat. Train the MIRA latent world model on a custom codec
# from scratch (no run.finetune_from). Auto-detects codec from codec_logs/ (trained
# via train_codec.sh), or falls back to frozen mira-mini codec if not found.
# The sibling mira repo (../mira) supplies scripts/configs and the pixi env.
#
# Env:
#   RUN     env prefix (default "pixi run --frozen"; "" if torch on PATH / venv active)
#   CODEC   custom codec .pth  (default: auto-detect from codec_logs/, fall back to frozen mira-mini)
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

# Custom codec from local training (codec_logs/), or fall back to frozen codec.
if [ -z "${CODEC:-}" ]; then
    # Priority 1: custom trained codec (from scratch)
    latest_codec=$(ls -d "$here/codec_logs/checkpoint-"*/ 2>/dev/null \
        | sed 's#.*/checkpoint-\([0-9]*\)/#\1#' | grep -xE '[0-9]+' | sort -n | tail -1)
    if [ -n "$latest_codec" ] && [ -f "$here/codec_logs/checkpoint-$latest_codec/checkpoint.pth" ]; then
        CODEC="$here/codec_logs/checkpoint-$latest_codec/checkpoint.pth"
        echo "Found custom codec (step $latest_codec), will use it"
    else
        # Priority 2: frozen codec from the downloaded mira-mini bundle
        for s in "$HOME"/.cache/huggingface/hub/models--alakazamworld--mira-mini/snapshots/*/; do
            [ -f "$s/codec/checkpoint-125000/checkpoint.pth" ] && CODEC="$s/codec/checkpoint-125000/checkpoint.pth"
        done
    fi
fi
[ -n "${CODEC:-}" ] && [ -f "$CODEC" ] || { echo "ERROR: codec not found. Train one first: ./train_codec.sh run.steps=125000 (or set CODEC=)"; exit 1; }

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
# Reduce CUDA fragmentation so the periodic world-model eval (DINO/Inception + rollout)
# fits in the headroom left by the resident training model (the 47 GB A40 is tight:
# ~36 GB training + ~10 GB eval). The OOM traceback explicitly recommends this.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
# Point the DINO loader at the local weights. mira/src/mira/codec/dino.py reads
# RS_DINO_WEIGHTS_DIR (NOT DINO_WEIGHTS_HOME); download_dino.sh stages them in
# $mira/dino_weights. Respect an existing override.
export RS_DINO_WEIGHTS_DIR="${RS_DINO_WEIGHTS_DIR:-$mira/dino_weights}"

# Fail fast if DINO weights missing: otherwise dino.py falls back to torch.hub
# (the HF DINOv3 mirrors are transformers-format, incompatible -> broken load),
# so training would run with unreliable/absent DINO metrics. finetune.sh inherits
# this guard via `exec train.sh`.
vitl16="$RS_DINO_WEIGHTS_DIR/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
vitb16="$RS_DINO_WEIGHTS_DIR/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"
if [ ! -f "$vitl16" ] || [ ! -f "$vitb16" ]; then
    echo "ERROR: DINO weights missing in $RS_DINO_WEIGHTS_DIR. Download first:"
    echo "  cd $here && ./download_dino.sh"
    exit 1
fi

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

# Checkpoints save as <output_dir>/checkpoint-<step>/checkpoint.pth (NOT
# <output_dir>/checkpoints/checkpoint-<step>.pth) -- glob THAT so resume actually
# fires. The old glob never matched, so runs never auto-resumed.
continue_from=""
latest_step=$(ls -d "$output_dir"/checkpoint-*/ 2>/dev/null \
    | sed 's#.*/checkpoint-\([0-9]*\)/#\1#' | grep -xE '[0-9]+' | sort -n | tail -1)
if [ -n "$latest_step" ] && [ -f "$output_dir/checkpoint-$latest_step/checkpoint.pth" ]; then
    ckpt_path="$output_dir/checkpoint-$latest_step/checkpoint.pth"
    continue_from="run.continue_from=$ckpt_path"
    echo "Auto-resuming from: $ckpt_path (step $latest_step)"
fi

echo "GPU free:"
nvidia-smi --query-gpu=memory.free --format=csv,noheader || true
echo "codec = $CODEC"
[ -n "${TRAIN_INDEX:-}" ] && { echo "train = $TRAIN_INDEX"; echo "test  = ${TEST_INDEX:-$TRAIN_INDEX}"; }
echo

exec $RUN python scripts/train_world_model.py \
    model.architecture.config.codec_checkpoint="$CODEC" \
    "${idx[@]}" \
    run.batch_size=1 run.compile=false wandb.mode=disabled dataloader.num_workers="$WORKERS" run.log_every=50 validation.downstream_val_every=7000 run.checkpoint_every=7000 run.checkpoint_keep_recent=2 \
    world_model_metrics.num_samples=32 world_model_metrics.dino_max_chunk_size=32 \
    run.output_dir="$output_dir" \
    $continue_from \
    '++tensorboard.logdir=${run.output_dir}/tb' "$@"
