#!/usr/bin/env bash
# Launch RacerX world-model training on LINUX (WSL or cluster). Wires the RacerX WebDataset
# (train + held-out test split) + the mira-mini frozen codec + optional warm-start, then calls
# the mira trainer (../mira/train.sh). Run from alakazam-mira-mini; the mira repo is the sibling.
#
# Env:
#   RX_ROOT   dir holding train/ and test/ WebDatasets   (required)
#               WSL:      /mnt/c/recordings/mira_wds
#               cluster:  wherever you rsync'd it
#   CODEC     .../mira-mini/.../codec/checkpoint-125000/checkpoint.pth   (required, frozen codec)
#   WM        .../mira-mini/.../checkpoint-52000/checkpoint.pth          (optional -> warm-start/finetune)
#   RUN       env prefix (default "pixi run --frozen"; "" if torch is already on PATH / venv active)
#
# Extra Hydra overrides pass through, e.g.:
#   RX_ROOT=/mnt/c/recordings/mira_wds CODEC=<codec.pth> ./train_racerx.sh run.steps=20000
#   WM=<wm.pth> RX_ROOT=... CODEC=... ./train_racerx.sh run.steps=5000     # finetune
set -uo pipefail
export HYDRA_FULL_ERROR=1
here="$(cd "$(dirname "$0")" && pwd)"
mira="$here/../mira"
[ -x "$mira/train.sh" ] || { echo "ERROR: mira trainer not found at $mira (clone it beside this repo)"; exit 1; }

: "${RX_ROOT:?set RX_ROOT=/path/to/mira_wds (holds train/ and test/)}"
export DATA_INDEX="$RX_ROOT/train/index.json"
[ -f "$DATA_INDEX" ] || { echo "ERROR: no train index at $DATA_INDEX -- build the WebDataset first"; exit 1; }
if [ -f "$RX_ROOT/test/index.json" ]; then
    export TEST_INDEX="$RX_ROOT/test/index.json"          # honest eval on the held-out split
    echo "data: train + held-out test ($RX_ROOT)"
else
    echo "data: train only ($RX_ROOT) -- no test split (eval reuses train; run holdout_wds.py for one)"
fi
: "${CODEC:?set CODEC=/path/to/mira-mini/codec/checkpoint-125000/checkpoint.pth}"
# WM is optional; train.sh warm-starts if it's set.

# dataloader.num_workers>0 is fine on Linux (unlike Windows); raise it for real runs via the args.
exec "$mira/train.sh" "$@"
