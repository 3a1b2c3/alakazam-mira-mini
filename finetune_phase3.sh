#!/usr/bin/env bash
# Phase 3: Finetune world model on RacerX data with new codec.
# Warm-start from wm_ckpt_98000.pth (converged baseline) with best Phase 1 codec.
# Conservative LR (5e-6) for stable finetuning. Delegates to train.sh.
#
# Usage:
#   ./finetune_phase3.sh                          # Default: 150k steps
#   ./finetune_phase3.sh run.steps=120000         # Shorter run
#   ./finetune_phase3.sh +optimizer.lr=1e-5       # Override LR
set -uo pipefail
export HYDRA_FULL_ERROR=1
here="$(cd "$(dirname "$0")" && pwd)"
mira="$here/../mira"

# Warm-start from best available checkpoint (latest from scratch training, fallback to outputs)
WM="${WM:-}"
if [ -z "$WM" ]; then
    latest_step=$(ls -d "$mira/train_world_model_logs_scratch"/checkpoint-*/ 2>/dev/null | sed 's#.*/checkpoint-\([0-9]*\)/#\1#' | grep -xE '[0-9]+' | sort -n | tail -1)
    [ -n "$latest_step" ] && [ -f "$mira/train_world_model_logs_scratch/checkpoint-$latest_step/checkpoint.pth" ] && WM="$mira/train_world_model_logs_scratch/checkpoint-$latest_step/checkpoint.pth"
    [ -z "$WM" ] && [ -f "$here/outputs/wm_ckpt_98000.pth" ] && WM="$here/outputs/wm_ckpt_98000.pth"
fi
[ -n "$WM" ] && [ -f "$WM" ] || { echo "ERROR: warm-start checkpoint not found"; exit 1; }
echo "Warm-starting from: $WM"

# Best codec from Phase 1: codec-81000 (26.05 dB PSNR, Aug 10 2026)
export CODEC="/home/horde/mira/codec_logs/checkpoint-81000/checkpoint.pth"

# Conservative finetuning settings
export OPT_DEFAULTS=(
    "+optimizer.lr=5e-6"
    "+optimizer.schedule=cosine_warmup"
    "+optimizer.warmup_steps=2000"
    "+model.grad_clip=1.0"
)

FT_DIR="${FT_DIR:-train_world_model_logs_scratch}"

finetune_arg=(run.finetune_from="$WM")
if ls "$mira/$FT_DIR"/checkpoint-*/checkpoint.pth >/dev/null 2>&1; then
    echo "Existing finetune checkpoint in $FT_DIR -> resuming; not re-warm-starting."
    finetune_arg=()
fi

exec "$here/train.sh" run.output_dir="$FT_DIR" "${finetune_arg[@]}" "$@"
