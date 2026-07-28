#!/usr/bin/env bash
# Upload the racer-x mira WebDataset straight to a Horde instance over SSH (rsync).
# Run from WSL/Linux (needs `rsync` + `ssh` access to the instance). Incremental --
# re-run as shards grow and only new/changed files transfer. Sends ONLY the shards
# (train/ + test/), not the wm_* / tb / wandb training outputs.
#
#   HOST      user@instance-ip        (required -- the Horde instance's SSH target)
#   DEST      remote mira_wds dir      (default /data/mira_wds)
#   RX_ROOT   local mira_wds dir       (default /mnt/c/recordings/mira_wds)
#   SSH_KEY   path to a private key    (optional -> ssh -i)
#
#   HOST=ubuntu@10.1.2.3 ./upload_to_horde.sh
#   HOST=... DEST=/mnt/data/mira_wds SSH_KEY=~/.ssh/horde ./upload_to_horde.sh
set -uo pipefail

: "${HOST:?set HOST=user@instance-ip (the Horde instance SSH target)}"
DEST="${DEST:-/data/mira_wds}"
RX_ROOT="${RX_ROOT:-/mnt/c/recordings/mira_wds}"
SSH_OPT=""
[ -n "${SSH_KEY:-}" ] && SSH_OPT="-i $SSH_KEY"

[ -f "$RX_ROOT/train/index.json" ] || { echo "ERROR: no data at $RX_ROOT/train -- build the WebDataset first"; exit 1; }

echo "== creating $DEST on $HOST =="
ssh $SSH_OPT "$HOST" "mkdir -p '$DEST'"

echo "== rsync $RX_ROOT/train -> $HOST:$DEST/ =="
rsync -avh --progress -e "ssh $SSH_OPT" "$RX_ROOT/train" "$HOST:$DEST/"
if [ -d "$RX_ROOT/test" ]; then
  echo "== rsync $RX_ROOT/test -> $HOST:$DEST/ =="
  rsync -avh --progress -e "ssh $SSH_OPT" "$RX_ROOT/test" "$HOST:$DEST/"
fi

echo
echo "Done. On the instance:  RX_ROOT=$DEST NPROC=8 ./train_horde.sh run.steps=20000"
