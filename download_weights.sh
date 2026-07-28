#!/usr/bin/env bash
# Linux mirror of download_weights.bat. Pre-download MIRA Mini weights into the HF
# cache so `mira-mini play` starts with no wait. Runs download_weights.py (same
# script the .bat uses) through the sibling mira env (pixi).
#   RUN   env prefix (default "pixi run --frozen"; "" if huggingface_hub on PATH)
# Usage:  ./download_weights.sh [1b|364m|all|<repo_id>]   (default: 1b)
set -uo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
mira="$here/../mira"
RUN="${RUN:-pixi run --frozen}"

WHICH="${1:-1b}"

echo "GPU (weights are large; make sure the disk has room):"
nvidia-smi --query-gpu=memory.free --format=csv,noheader || true
echo
cd "$mira"
exec $RUN python "$here/download_weights.py" "$WHICH"
