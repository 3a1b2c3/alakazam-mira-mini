#!/usr/bin/env bash
# Linux mirror of finetune_racerx.bat. FINETUNE the RacerX world model by
# warm-starting from the mira-mini 1B checkpoint (checkpoint-52000), on the frozen
# mira-mini codec, with honest eval on the held-out test split. Recommended over
# from-scratch at limited data. Nothing hardcoded: data root is a parameter; codec
# + WM are globbed from the HF cache.
#
#   RX_ROOT   mira_wds dir (holds train/ + test/); default /mnt/c/recordings/mira_wds
#   RUN       env prefix (default "pixi run --frozen"; "" if torch on PATH)
#   WORKERS   dataloader workers (default 4 -- Linux has NCCL + fast workers)
#   $1        steps (default 10000);  extra Hydra overrides pass through
# Usage:
#   ./finetune_racerx.sh                 10000 steps, default data root
#   ./finetune_racerx.sh 20000
#   RX_ROOT=/data/mira_wds ./finetune_racerx.sh 10000 run.batch_size=2
set -uo pipefail
export HYDRA_FULL_ERROR=1
here="$(cd "$(dirname "$0")" && pwd)"
mira="$here/../mira"
[ -f "$mira/pixi.toml" ] || { echo "ERROR: mira trainer not found at $mira (clone it beside this repo)"; exit 1; }
RUN="${RUN:-pixi run --frozen}"
WORKERS="${WORKERS:-4}"

REC="${RX_ROOT:-/mnt/c/recordings/mira_wds}"
[ -f "$REC/train/index.json" ] || { echo "ERROR: no RacerX train index at $REC/train -- build the WebDataset first"; exit 1; }
TESTIDX="$REC/train/index.json"
if [ -f "$REC/test/index.json" ]; then TESTIDX="$REC/test/index.json"; echo "eval on held-out test split"; else echo "NOTE: no test split -- eval reuses train (run holdout_wds.py)"; fi

# frozen codec + warm-start WM from the mira-mini HF snapshot (globbed -> no hardcoded hash)
SNAP=""
for s in "$HOME"/.cache/huggingface/hub/models--alakazamworld--mira-mini/snapshots/*/; do
    [ -f "$s/codec/checkpoint-125000/checkpoint.pth" ] && SNAP="$s"
done
[ -n "$SNAP" ] || { echo "ERROR: mira-mini weights not found -- run ./download_weights.sh 1b"; exit 1; }
CODEC="${CODEC:-$SNAP/codec/checkpoint-125000/checkpoint.pth}"
WM="${WM:-$SNAP/checkpoint-52000/checkpoint.pth}"
[ -f "$WM" ] || { echo "ERROR: warm-start checkpoint-52000 not found in $SNAP"; exit 1; }

# $1 is the step count ONLY if it's all digits; a key=value first arg is a Hydra
# override (e.g. tensorboard.logdir=...), kept in "$@" instead of becoming run.steps.
STEPS=50000
case "${1:-}" in
    ''|*[!0-9]*) : ;;            # empty / non-numeric -> keep default, don't consume
    *) STEPS="$1"; shift ;;      # pure digits -> use as steps
esac

# AUTO-RESUME: if a checkpoint already exists in the output dir, CONTINUE from it
# (restores optimizer + step counter -> picks up where a killed run left off); else
# warm-start from the mira-mini WM. keep_recent=1 -> at most one checkpoint dir.
OUTDIR="${OUT:-$REC/wm_racerx_ft}"
CKPT="$(ls -d "$OUTDIR"/checkpoint-*/ 2>/dev/null | sort -V | tail -1)"
if [ -n "$CKPT" ] && [ -f "${CKPT}checkpoint.pth" ]; then
    STARTARG="run.continue_from=${CKPT}checkpoint.pth"; echo "mode: RESUME from ${CKPT}checkpoint.pth"
else
    STARTARG="run.finetune_from=$WM"; echo "mode: FINETUNE (warm-start) from $WM"
fi

# NOTE: Linux HAS NCCL -- the Windows LOCAL_RANK/RANK/... clearing is intentionally omitted.
cd "$mira"
echo "GPU free:"
nvidia-smi --query-gpu=memory.free --format=csv,noheader || true
echo "codec      = $CODEC"
echo "start      = $STARTARG"
echo "train      = $REC/train/index.json"
echo "test       = $TESTIDX"
echo "output     = $OUTDIR  ($STEPS steps)"
echo

# Uses the FINETUNE config (finetune_world_model.yaml): lr 5e-5, log_every 50, val_every 3000,
# shuffle 1000, checkpoint retention. Only launcher-specific overrides remain below.
exec $RUN python scripts/train_world_model.py --config-name finetune_world_model \
    model.architecture.config.codec_checkpoint="$CODEC" "$STARTARG" \
    dataset.train_index="$REC/train/index.json" dataset.test_index="$TESTIDX" \
    run.batch_size=1 run.compile=true wandb.mode=disabled dataloader.num_workers="$WORKERS" \
    run.steps="$STEPS" run.output_dir="$OUTDIR" run.checkpoint_every=250 optim.scheduler.warmup_steps=200 \
    validation.val_n_samples=64 world_model_metrics.num_samples=128 \
    ++tensorboard.logdir="$OUTDIR/tb" "$@"
