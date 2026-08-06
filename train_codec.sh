#!/usr/bin/env bash
# Train the MIRA RAEv2 video codec from scratch (step 1, no frozen anything).
# Codec is the foundational stage before training the latent world model.
#
# Env:
#   RUN     env prefix (default "pixi run --frozen"; "" if torch on PATH / venv active)
#   TRAIN_INDEX / TEST_INDEX   data indices (default: sourced from mira/data_paths.sh
#                              written by get_data.sh; else fall back to
#                              $MIRA_WDS/{train,test}/index.json).
#   MIRA_WDS  local WebDataset root (default /home/horde/mira_wds)
#   WORKERS dataloader workers (default 4)
#
# Usage:
#   ./train_codec.sh                    # Default: codec_logs
#   ./train_codec.sh run.steps=200      # Custom steps
#   ./train_codec.sh run.batch_size=2   # Custom batch size
#
set -uo pipefail
export HYDRA_FULL_ERROR=1
here="$(cd "$(dirname "$0")" && pwd)"
mira="$here/../mira"
[ -f "$mira/pixi.toml" ] || { echo "ERROR: mira trainer not found at $mira (clone it beside this repo)"; exit 1; }
RUN="${RUN:-pixi run}"
WORKERS="${WORKERS:-4}"

# Default data indices: sourced from data_paths.sh (get_data.sh) unless already set.
[ -z "${TRAIN_INDEX:-}" ] && [ -f "$mira/data_paths.sh" ] && . "$mira/data_paths.sh"
# Fallback: the local WebDataset at $MIRA_WDS (default /home/horde/mira_wds).
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

cd "$mira"
export WANDB_MODE=offline
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export RS_DINO_WEIGHTS_DIR="${RS_DINO_WEIGHTS_DIR:-$mira/dino_weights}"

# Codec training uses DINO for latent consistency loss
vitl16="$RS_DINO_WEIGHTS_DIR/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
vitb16="$RS_DINO_WEIGHTS_DIR/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"
if [ ! -f "$vitl16" ] || [ ! -f "$vitb16" ]; then
    echo "ERROR: DINO weights missing in $RS_DINO_WEIGHTS_DIR. Download first:"
    echo "  cd $here && ./download_dino.sh"
    exit 1
fi

# Auto-detect and resume from latest checkpoint if it exists
output_dir="codec_logs"
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
[ -n "${TRAIN_INDEX:-}" ] && { echo "train = $TRAIN_INDEX"; echo "test  = ${TEST_INDEX:-$TRAIN_INDEX}"; }
echo

# Start background PSNR monitor (evaluates each checkpoint)
{
    declare -A evaluated
    cd "$here"  # Stay in alakazam-mira-mini directory
    while true; do
        latest=$(ls -d codec_logs/checkpoint-*/checkpoint.pth 2>/dev/null | sed 's#.*/checkpoint-\([0-9]*\)/.*#\1#' | sort -n | tail -1)
        if [ -n "$latest" ] && [ -z "${evaluated[$latest]:-}" ]; then
            echo "[$(date '+%H:%M:%S')] Evaluating codec checkpoint-$latest..."
            if python codec_recon_psnr.py --checkpoint "codec_logs/checkpoint-$latest/checkpoint.pth" --num-samples 16 2>&1 | tee -a codec_psnr_log.txt; then
                evaluated[$latest]=1
            fi
        fi
        sleep 5
    done
} &
MONITOR_PID=$!
trap "kill $MONITOR_PID 2>/dev/null" EXIT

# Optimizer defaults (prevent divergence at step 25k->30k):
OPT_DEFAULTS=(
    "+optimizer.lr=1e-4"
    "+optimizer.schedule=cosine_warmup"
    "+optimizer.warmup_steps=3000"
    "+model.grad_clip=1.0"
    "++model.loss.weights.dino_latent_consistency_frame_frac=0.1"
)
# Override examples:
#   ./train_codec.sh +optimizer.lr=5e-5          (lower LR if diverging)
#   ./train_codec.sh run.steps=100000            (longer training, better quality)
#   ./train_codec.sh +optimizer.lr=5e-4          (more aggressive learning)

exec $RUN python scripts/train_codec.py \
    "${idx[@]}" \
    "${OPT_DEFAULTS[@]}" \
    run.batch_size=2 run.compile=false wandb.mode=disabled dataloader.num_workers="$WORKERS" run.log_every=50 run.checkpoint_every=3000 run.checkpoint_keep_recent=2 validation.val_every=2500 \
    run.output_dir="codec_logs" \
    $continue_from \
    '++tensorboard.logdir=${run.output_dir}/tb' "$@"
