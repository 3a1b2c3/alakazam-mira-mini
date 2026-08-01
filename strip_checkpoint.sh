#!/usr/bin/env bash
# Strip optimizer state from checkpoint (29 GB -> ~5-10 GB for inference/archival).
# Run on Horde to reduce checkpoint size before syncing to Windows.
# Usage: ./strip_checkpoint.sh <checkpoint-dir>
#   e.g.: ./strip_checkpoint.sh ~/mira_wds/wm_racerx_ft/checkpoint-24500

set -uo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
mira="$here/../mira"

ckpt_dir="$1"
if [ ! -f "$ckpt_dir/checkpoint.pth" ]; then
    echo "ERROR: checkpoint.pth not found in $ckpt_dir"
    exit 1
fi

cd "$mira"
pixi run python "$here/strip_checkpoint.py" "$ckpt_dir/checkpoint.pth" --output "$ckpt_dir/checkpoint-weights-only.pth"
