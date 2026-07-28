#!/usr/bin/env bash
# Linux mirror of tensorboard.bat. Launch TensorBoard for MIRA training runs. Reads
# the scalar logs the trainer writes (tensorboard.logdir=${run.output_dir}/tb, ON by
# default). Points at the parent mira_wds so every run dir (e.g. wm_racerx_ft/tb)
# shows up as a separate TensorBoard run. Served via the sibling mira pixi env.
#   REC   default logdir root (default /mnt/c/recordings/mira_wds; env override)
#   RUN   env prefix (default "pixi run --frozen"; "" if tensorboard on PATH)
# Usage:
#   ./tensorboard.sh                 -> serve $REC on :6006
#   ./tensorboard.sh <logdir>        custom logdir
#   ./tensorboard.sh <logdir> <port> custom logdir + port
set -uo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
mira="$here/../mira"
RUN="${RUN:-pixi run --frozen}"

LOGDIR="${1:-${REC:-/mnt/c/recordings/mira_wds}}"
PORT="${2:-6006}"

[ -d "$LOGDIR" ] || { echo "ERROR: logdir not found: $LOGDIR  (has training written any TB events yet?)"; exit 1; }

cd "$mira"
echo "Serving TensorBoard for \"$LOGDIR\" on http://localhost:$PORT  (Ctrl+C to stop)"
exec $RUN tensorboard --logdir "$LOGDIR" --port "$PORT"
