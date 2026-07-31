#!/usr/bin/env bash
# Linux mirror of download_models.bat. Download MIRA data/models via the sibling
# mira env (pixi). Downloads the rocket-science DATASET (HuggingFace
# kyutai/rocket-science): 2v2 clips + actions + game state. No pretrained MIRA
# checkpoints are published (train from scratch with scripts/train_*.py).
#
# Knobs:  SPLIT=test|train   SHARDS=1  (1 = one shard/quick; 0 = full split)
#         RUN               env prefix (default "pixi run --frozen"; "" if torch on PATH)
# Usage:  ./download_models.sh
set -uo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
mira="$here/../mira"
[ -f "$mira/pixi.toml" ] || { echo "ERROR: mira trainer not found at $mira (clone it beside this repo)"; exit 1; }

RUN="${RUN:-pixi run --frozen}"
SPLIT="${SPLIT:-test}"
SHARDS="${SHARDS:-1}"
# Disable the Xet backend: on this Horde box xet_get() dies with
# "Unable to parse string as hex hash value" mid-shard. Force plain HTTP.
# Override with HF_HUB_DISABLE_XET=0 to re-enable if xet ever gets fixed.
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"

echo "--- downloading rocket-science: split=$SPLIT shards=$SHARDS (0=full) ---"
cd "$mira"
$RUN python -c "from mira.data import RocketScienceDataset; ds=RocketScienceDataset.from_hub('kyutai/rocket-science', split='$SPLIT', shards=($SHARDS or None)); print('downloaded', len(ds.match_ids()), 'matches for split $SPLIT')" \
    || { echo "ERROR: dataset download failed"; exit 1; }

echo
echo "=== done: rocket-science ($SPLIT) cached ==="
echo
echo "NOTE: DINOv3-L/16 is only needed for CODEC TRAINING (not world-model train/inference)."
echo "It is gated by Meta (dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth). If you have access,"
echo "download it with ./download_dino.sh (DINOV3_VITL16_URL) or set RS_DINO_WEIGHTS_DIR=/path/to/weights."
