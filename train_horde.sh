#!/usr/bin/env bash
# Multi-GPU MIRA world-model training on a Horde / Linux GPU box (NO SLURM), via pixi.
# Installs pixi + the locked linux-64 env in the sibling mira trainer (../mira), fetches the
# mira-mini codec + 1B checkpoint (unless CODEC/WM are set), and torchruns the finetune across
# all GPUs. Run from alakazam-mira-mini; the mira repo must be a sibling (../mira).
#
# Env:
#   NPROC     GPUs on this box                         (default 8)
#   RX_ROOT   dir holding train/index.json (+ test/)   (required)
#   CODEC     frozen codec .pth      (default: auto-fetch from mira-mini)
#   WM        warm-start 1B .pth     (default: auto-fetch; ignored if SCRATCH=1)
#   SCRATCH   1 = train WM from scratch (no warm-start) (default 0 = finetune)
#   STEPS BATCH WORKERS HF_REPO                          (tunable)
#
#   NPROC=8 RX_ROOT=/data/mira_wds ./train_horde.sh run.steps=20000
#   NPROC=8 RX_ROOT=... SCRATCH=1 STEPS=100000 ./train_horde.sh          # from scratch
set -uo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
mira="$here/../mira"
[ -f "$mira/pixi.toml" ] || { echo "ERROR: mira trainer (with pixi.toml) not found at $mira -- clone it beside this repo"; exit 1; }

NPROC="${NPROC:-8}"
STEPS="${STEPS:-20000}"
BATCH="${BATCH:-2}"                  # per-GPU; effective batch = BATCH * NPROC
WORKERS="${WORKERS:-1}"             # keep low: shards-per-rank is small
HF_REPO="${HF_REPO:-alakazamworld/mira-mini}"

# ---- data source: provide EITHER a local RX_ROOT (dir with train/index.json) OR HF_DATA_REPO
# to pull the racer-x WebDataset from a (private) HF dataset -- the Horde-friendly path. Upload it
# once from your box with alakazam-mira-mini/upload_racerx_data.bat, re-run as shards grow.
DATA_REPO="${HF_DATA_REPO:-}"

# ---- pixi + locked env (pixi.lock is linux-64; lives in the mira repo) ------
command -v pixi >/dev/null 2>&1 || { echo "== install pixi =="; curl -fsSL https://pixi.sh/install.sh | bash; }
export PATH="$HOME/.pixi/bin:$PATH"
cd "$mira"
echo "== pixi install --locked =="
pixi install --locked

# ---- data: local RX_ROOT if it has the shards, else pull HF_DATA_REPO (needs pixi for HF) ----
if [ -z "${RX_ROOT:-}" ] || [ ! -f "$RX_ROOT/train/index.json" ]; then
  [ -n "$DATA_REPO" ] || { echo "ERROR: set RX_ROOT=/path/to/mira_wds (with train/index.json), or HF_DATA_REPO=user/racerx-mira-wds to pull it"; exit 1; }
  echo "== fetch racer-x WebDataset from HF dataset $DATA_REPO =="
  RX_ROOT="$(pixi run python - <<PY
from huggingface_hub import snapshot_download
print(snapshot_download("$DATA_REPO", repo_type="dataset"))
PY
)"
fi
TRAIN="$RX_ROOT/train/index.json"
[ -f "$TRAIN" ] || { echo "ERROR: no train index at $TRAIN after data fetch"; exit 1; }
TEST="$RX_ROOT/test/index.json"; [ -f "$TEST" ] || TEST="$TRAIN"   # reuse train if no held-out split

# ---- checkpoints: CODEC/WM if set, else fetch the mira-mini bundle ----------
if [ -z "${CODEC:-}" ] || { [ "${SCRATCH:-0}" != "1" ] && [ -z "${WM:-}" ]; }; then
  echo "== fetch $HF_REPO (codec + 1B) =="
  SNAP="$(pixi run python - <<PY
from huggingface_hub import snapshot_download
print(snapshot_download("$HF_REPO", allow_patterns=["codec/checkpoint-125000/*", "checkpoint-52000/*"]))
PY
)"
  CODEC="${CODEC:-$SNAP/codec/checkpoint-125000/checkpoint.pth}"
  [ "${SCRATCH:-0}" = "1" ] || WM="${WM:-$SNAP/checkpoint-52000/checkpoint.pth}"
fi
[ -f "$CODEC" ] || { echo "ERROR: codec not found: $CODEC"; exit 1; }

# AUTO-RESUME: if a checkpoint already exists in the output dir, CONTINUE from it (restores
# optimizer + step counter -> picks up a killed/pre-empted run); else warm-start from WM, or
# start from scratch if SCRATCH=1. keep_recent=1 -> at most one checkpoint dir.
OUT="${OUT:-$RX_ROOT/wm_racerx_ft}"
CKPT="$(ls -d "$OUT"/checkpoint-*/ 2>/dev/null | sort -V | tail -1)"
if [ -n "$CKPT" ] && [ -f "${CKPT}checkpoint.pth" ]; then
  START=(run.continue_from="${CKPT}checkpoint.pth"); echo "mode: RESUME from ${CKPT}checkpoint.pth"
elif [ "${SCRATCH:-0}" = "1" ]; then
  START=(); echo "mode: FROM SCRATCH (random-init DiT on the frozen codec)"
else
  [ -f "${WM:-}" ] || { echo "ERROR: warm-start ckpt not found: ${WM:-<unset>} (set SCRATCH=1 to train from scratch)"; exit 1; }
  START=(run.finetune_from="$WM"); echo "mode: FINETUNE from $WM"
fi

# ---- launch (torchrun fans out to all GPUs via NCCL; pixi supplies the env) -
echo "== GPUs =="; nvidia-smi --query-gpu=name,memory.free --format=csv,noheader
echo "== torchrun --nproc_per_node=$NPROC  ($STEPS steps, batch ${BATCH}x${NPROC})  data=$RX_ROOT =="
exec pixi run torchrun --nproc_per_node="$NPROC" scripts/train_world_model.py \
  model.architecture.config.codec_checkpoint="$CODEC" \
  dataset.train_index="$TRAIN" dataset.test_index="$TEST" \
  run.batch_size="$BATCH" run.steps="$STEPS" run.output_dir="$OUT" \
  dataloader.num_workers="$WORKERS" wandb.mode=offline \
  "${START[@]}" "$@"
