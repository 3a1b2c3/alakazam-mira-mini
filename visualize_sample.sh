#!/usr/bin/env bash
# 3D visualize a sample from MIRA WebDataset on Horde.
# Usage: ./visualize_sample.sh [--sample 0] [--output out.png]
set -uo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
mira="$here/../mira"
cd "$mira"
pixi run python "$here/visualize_sample.py" --data ~/mira_wds/train/index.json "$@"
