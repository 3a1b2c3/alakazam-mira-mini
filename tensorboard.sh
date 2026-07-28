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

# default logdir: explicit arg > REC env > first existing home/Windows candidate
LOGDIR="${1:-${REC:-}}"
if [ -z "$LOGDIR" ]; then
    for c in "$HOME/mira_wds/wm_racerx_ft" "$HOME/mira_wds" /mnt/c/recordings/mira_wds; do
        [ -d "$c" ] && { LOGDIR="$c"; break; }
    done
    LOGDIR="${LOGDIR:-/mnt/c/recordings/mira_wds}"
fi
PORT="${2:-6006}"

[ -d "$LOGDIR" ] || { echo "ERROR: logdir not found: $LOGDIR  (has training written any TB events yet?)"; exit 1; }

cd "$mira"
echo "Serving TensorBoard for \"$LOGDIR\" on http://localhost:$PORT  (Ctrl+C to stop)"
exec $RUN tensorboard --logdir "$LOGDIR" --port "$PORT"
