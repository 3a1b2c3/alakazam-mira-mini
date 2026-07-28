#!/usr/bin/env bash
# Linux smoke suite: every runnable MIRA training path (data test, codec from scratch,
# WM finetune, WM from scratch, full codec->WM chain), each capped to reach a checkpoint
# in minutes. Reports PASS/FAIL by verifying each run WROTE a checkpoint (exit 0 alone
# isn't enough -- a run can 0-exit after a stalled validation). Runs all steps even if
# one fails. Uses the train.sh / train_codec.sh wrappers. For the cluster or WSL.
#
# Env (paths -- nothing hardcoded):
#   DATA_INDEX           /path/to/mira_wds/train/index.json        (required)
#   OUT                  /path/to/output/root (runs write under it) (required)
#   CODEC                .../codec/checkpoint-125000/checkpoint.pth (WM steps)
#   WM                   .../checkpoint-52000/checkpoint.pth        (finetune step)
#   RS_DINO_WEIGHTS_DIR  dir with dinov3_vitl16_...pth              (codec step)
#   RUN                  env prefix (default "pixi run --frozen")
# Usage:  ./smoke_test_all.sh [steps]          (default 20)
set -uo pipefail
export HYDRA_FULL_ERROR=1
cd "$(dirname "$0")"

STEPS="${1:-20}"
CAPS="validation.val_n_samples=8 world_model_metrics.num_samples=16"
export RUN="${RUN:-pixi run --frozen}"

: "${DATA_INDEX:?set DATA_INDEX=/path/to/train/index.json}"
: "${OUT:?set OUT=/path/to/output/root}"

ckpt() { for d in "$1"/checkpoint-*; do [ -f "$d/checkpoint.pth" ] && { echo PASS; return; }; done; echo FAIL; }

declare -A R
echo "=== GPU free ==="; nvidia-smi --query-gpu=memory.free,memory.used --format=csv,noheader || true; echo

echo "############### 1/5 data loader test ###############"
if $RUN python scripts/test_mira_dataset.py --data "$(dirname "$DATA_INDEX")"; then R[1]=PASS; else R[1]=FAIL; fi

echo "############### 2/5 codec from scratch ###############"
if [ -z "${RS_DINO_WEIGHTS_DIR:-}" ]; then echo "SKIP: set RS_DINO_WEIGHTS_DIR"; R[2]=SKIP
else ./train_codec.sh run.steps="$STEPS" validation.val_n_samples=8 run.output_dir="$OUT/codec_smoke" || true; R[2]=$(ckpt "$OUT/codec_smoke"); fi

echo "############### 3/5 world model FINETUNE ###############"
if [ -z "${CODEC:-}" ] || [ -z "${WM:-}" ]; then echo "SKIP: set CODEC= and WM="; R[3]=SKIP
else ./train.sh run.steps="$STEPS" $CAPS run.output_dir="$OUT/wm_finetune_smoke" || true; R[3]=$(ckpt "$OUT/wm_finetune_smoke"); fi

echo "############### 4/5 world model FROM SCRATCH ###############"
if [ -z "${CODEC:-}" ]; then echo "SKIP: set CODEC="; R[4]=SKIP
else ( unset WM; ./train.sh run.steps="$STEPS" $CAPS run.output_dir="$OUT/wm_scratch_smoke" ) || true; R[4]=$(ckpt "$OUT/wm_scratch_smoke"); fi

echo "############### 5/5 full chain (step-2 codec -> WM) ###############"
fresh=""; for d in "$OUT"/codec_smoke/checkpoint-*; do [ -f "$d/checkpoint.pth" ] && fresh="$d/checkpoint.pth"; done
if [ -z "$fresh" ]; then echo "SKIP: no codec checkpoint to chain from"; R[5]=SKIP
else ( unset WM; CODEC="$fresh" ./train.sh run.steps="$STEPS" $CAPS run.output_dir="$OUT/wm_chain_smoke" ) || true; R[5]=$(ckpt "$OUT/wm_chain_smoke"); fi

echo
echo "==================== SMOKE TEST SUMMARY (PASS = wrote a checkpoint) ===================="
printf "  1 data loader test ......... %s\n" "${R[1]:-?}"
printf "  2 codec from scratch ....... %s\n" "${R[2]:-?}"
printf "  3 world model finetune ..... %s\n" "${R[3]:-?}"
printf "  4 world model from scratch . %s\n" "${R[4]:-?}"
printf "  5 full chain (codec->WM) ... %s\n" "${R[5]:-?}"
echo "  - SLURM (train.sbatch) ....... run this suite via sbatch to cover it"
echo "======================================================================================="
