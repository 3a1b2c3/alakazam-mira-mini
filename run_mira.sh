#!/usr/bin/env bash
# Linux mirror of run_mira.bat. Launch MIRA Mini locally in the browser via the
# installed `mira-mini` entry point. First run downloads the weights on demand (or
# run ./download_weights.sh first). --model auto picks the 1b on CUDA.
#   MIRA_MINI  path to the mira-mini executable (default: `mira-mini` on PATH)
#   RUN        env prefix (default ""; e.g. "pixi run" if it lives in a pixi env)
# Usage:  ./run_mira.sh [extra args]   e.g.  ./run_mira.sh --model 364m --steps 8
set -uo pipefail
RUN="${RUN:-}"
MIRA_MINI="${MIRA_MINI:-mira-mini}"
$RUN command -v "$MIRA_MINI" >/dev/null 2>&1 || { echo "ERROR: '$MIRA_MINI' not found -- install alakazam-mira-mini into your env (set MIRA_MINI= or RUN=)"; exit 1; }

echo "GPU before launch:"
nvidia-smi --query-gpu=memory.total,memory.used,memory.free --format=csv,noheader || true
echo
echo "Starting MIRA Mini play - a browser tab will open when it is ready..."
# Default to the 1b model (discrete GPU). A later --model on the command line overrides it.
exec $RUN "$MIRA_MINI" play --model 1b "$@"
