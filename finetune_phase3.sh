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

# Warm-start from proven checkpoint (98k, before 112k degradation)
WM="${WM:-$here/outputs/wm_ckpt_98000.pth}"
[ -f "$WM" ] || { echo "ERROR: warm-start checkpoint not found: $WM"; exit 1; }

# Best codec from Phase 1 (update after evaluation determines peak)
export CODEC="/home/horde/mira/codec_logs/checkpoint-40000/checkpoint.pth"

# Conservative finetuning settings
export OPT_DEFAULTS=(
    "+optimizer.lr=5e-6"
    "+optimizer.schedule=cosine_warmup"
    "+optimizer.warmup_steps=2000"
    "+model.grad_clip=1.0"
)

mira="$here/../mira"
FT_DIR="${FT_DIR:-train_world_model_logs_scratch}"

finetune_arg=(run.finetune_from="$WM")
if ls "$mira/$FT_DIR"/checkpoint-*/checkpoint.pth >/dev/null 2>&1; then
    echo "Existing finetune checkpoint in $FT_DIR -> resuming; not re-warm-starting."
    finetune_arg=()
fi

exec "$here/train.sh" run.output_dir="$FT_DIR" "${finetune_arg[@]}" "$@"
