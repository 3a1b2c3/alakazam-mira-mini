#!/usr/bin/env bash
# Linux mirror of eval_wm.bat. Offline-evaluate a world-model checkpoint (metrics
# on a saved checkpoint, no training) via mira's scripts/eval_world_model_offline.py,
# which reads the run's world_model_config.yaml saved 2 dirs above the .pth so the
# eval matches how the model was trained. Single-process (no torchrun).
#
# Reports: validation loss + world-model metrics (DINO/latent drift, Frechet
# DINO/Inception = gFDD/gFID-analogs). Capped small by default for speed -- raise
# --num-samples / --val-n-samples for a real (slow) eval.
#
#   REC                  mira_wds root (holds wm_* runs); default /mnt/c/recordings/mira_wds
#   RS_DINO_WEIGHTS_DIR  dir with dinov3_vitb16_...pth (default: mira/dino_weights)
#   RUN                  env prefix (default "pixi run --frozen"; "" if torch on PATH)
# Usage:
#   ./eval_wm.sh                          DEFAULT: newest local RacerX WM checkpoint
#   ./eval_wm.sh <checkpoint.pth>         a specific checkpoint
#   ./eval_wm.sh <ckpt> --num-samples 256 --val-n-samples 128     (fuller eval)
#   ./eval_wm.sh <ckpt> --viz 4                                    (also render 4 rollouts)
#   ./eval_wm.sh --viz 4                                           (default ckpt + extra args)
set -uo pipefail
export HYDRA_FULL_ERROR=1
here="$(cd "$(dirname "$0")" && pwd)"
mira="$here/../mira"
[ -f "$mira/pixi.toml" ] || { echo "ERROR: mira trainer not found at $mira (clone it beside this repo)"; exit 1; }
RUN="${RUN:-pixi run --frozen}"
REC="${REC:-/mnt/c/recordings/mira_wds}"

# First arg is the checkpoint unless it's a flag (starts with '-'); else use default.
CKPT=""
if [ "$#" -gt 0 ] && [ "${1#-}" = "$1" ]; then CKPT="$1"; shift; fi

# DEFAULT = newest WM checkpoint from a LOCAL RacerX run = highest checkpoint-<N>
# under $REC/wm_*. 'checkpoint-*' skips in-progress '.checkpoint-N.tmp' dirs; 'wm_*'
# skips the codec. NOTE: the released reference (checkpoint-52000) can NOT be the
# default -- it's a Rocket League model (video.timesteps=80) that fails on RacerX's
# 80-frame chunks; only RacerX-trained checkpoints eval on this data.
if [ -z "$CKPT" ]; then
    best=-1
    for d in "$REC"/wm_*/checkpoint-*; do
        [ -f "$d/checkpoint.pth" ] || continue
        n="${d##*/checkpoint-}"
        case "$n" in ''|*[!0-9]*) continue;; esac
        if [ "$n" -gt "$best" ]; then best="$n"; CKPT="$d/checkpoint.pth"; fi
    done
    [ -n "$CKPT" ] && echo "default checkpoint (newest local RacerX WM): $CKPT"
fi
[ -n "$CKPT" ] || { echo "ERROR: no WM checkpoint under $REC/wm_* -- run a training first, or pass a path."; exit 1; }
[ -f "$CKPT" ] || { echo "ERROR: checkpoint not found: $CKPT"; exit 1; }

# World-model metrics load DINOv3-base (vitb16) via torch.hub for the DINO-Frechet
# metric; point it at local weights.
RS_DINO_WEIGHTS_DIR="${RS_DINO_WEIGHTS_DIR:-$mira/dino_weights}"
export RS_DINO_WEIGHTS_DIR

# NOTE: Linux HAS NCCL -- the Windows LOCAL_RANK/RANK/... clearing is intentionally omitted.
cd "$mira"
echo "GPU free:"
nvidia-smi --query-gpu=memory.free --format=csv,noheader || true
echo "eval = $CKPT"
echo

# torch.hub crashes on a Path(None) if the base weights aren't present. Auto-skip
# the metrics (validation-loss eval still runs) instead of crashing.
VITB16="$RS_DINO_WEIGHTS_DIR/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"
AUTOSKIP=()
if [ ! -f "$VITB16" ]; then
    echo "WARNING: DINOv3-base weights not found:"
    echo "         $VITB16"
    echo "         world-model metrics need them -- running VALIDATION LOSS ONLY."
    echo "         Run ./download_dino.sh (with DINOV3_VITB16_URL) to enable full metrics."
    echo
    AUTOSKIP=(--skip-metrics)
fi

# Eval on the RacerX WebDataset when its index exists (released checkpoints bake the
# author's absolute data paths into their config; --test-index overrides that). A
# user --test-index in the extra args overrides this (last wins).
IDXARG=()
[ -f "$REC/train/index.json" ] && IDXARG=(--test-index "$REC/train/index.json")

# small caps by default so a smoke checkpoint evals in minutes; override by passing
# your own --num-samples / --val-n-samples in the extra args (last wins).
exec $RUN python scripts/eval_world_model_offline.py "$CKPT" "${IDXARG[@]}" \
    --num-samples 16 --val-n-samples 8 "${AUTOSKIP[@]}" "$@"
