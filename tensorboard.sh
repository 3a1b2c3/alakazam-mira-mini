#!/usr/bin/env bash
# Linux mirror of tensorboard.bat -- run ON the horde box to serve mira's training
# TensorBoard, then view it locally through an SSH tunnel:
#   ssh -L 6006:localhost:6006 <horde-host>      # then open http://localhost:6006
# Reads the scalar logs the trainer writes (++tensorboard.logdir=${run.output_dir}/tb,
# ON by default). Defaults to the parent mira dir so EVERY run
# (train_world_model_logs*/tb: scratch, finetune, ...) shows up as a separate run.
#   RUN   unused here (TensorBoard only reads events; see note below)
# Usage:
#   ./tensorboard.sh                 -> serve $mira on :6006
#   ./tensorboard.sh <logdir>        custom logdir
#   ./tensorboard.sh <logdir> <port> custom logdir + port
set -uo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
mira="$here/../mira"

LOGDIR="${1:-${LOGDIR:-$mira}}"
PORT="${2:-6006}"

[ -d "$LOGDIR" ] || { echo "ERROR: logdir not found: $LOGDIR  (has training written any TB events yet?)"; exit 1; }

# TensorBoard only READS event files -- it doesn't need the pixi training env (whose
# python has no pip anyway). Use the system python3 + user site; auto-install if missing.
export PATH="$HOME/.local/bin:$PATH"
if ! command -v tensorboard >/dev/null 2>&1 && ! python3 -c "import tensorboard" >/dev/null 2>&1; then
    echo "== installing tensorboard (system python3, --user, one-time) =="
    python3 -m pip install --user tensorboard || pip install --user tensorboard || {
        echo "ERROR: could not install tensorboard"; exit 1; }
fi
echo "Serving TensorBoard for \"$LOGDIR\" on http://localhost:$PORT  (Ctrl+C to stop)"
echo "  tunnel from your machine:  ssh -L $PORT:localhost:$PORT <horde-host>"
if command -v tensorboard >/dev/null 2>&1; then
    exec tensorboard --logdir "$LOGDIR" --port "$PORT"
else
    exec python3 -m tensorboard.main --logdir "$LOGDIR" --port "$PORT"
fi
