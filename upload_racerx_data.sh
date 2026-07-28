#!/usr/bin/env bash
# Linux mirror of upload_racerx_data.bat. Push the racer-x mira WebDataset
# (RX_ROOT/{train,test}) to a PRIVATE HuggingFace dataset so Horde instances can
# pull it (train_horde.sh HF_DATA_REPO=...). Re-run as shards grow -- upload_folder
# is incremental (only changed/new files transfer). Uploads ONLY the shards, not the
# wm_* / tb / wandb training outputs.
#
# Needs an HF token with write access: `pixi run huggingface-cli login` or HF_TOKEN.
#
#   HF_DATA_REPO   youruser/racerx-mira-wds   (required)
#   RX_ROOT        mira_wds dir                (default /mnt/c/recordings/mira_wds)
#   RUN            env prefix (default "pixi run --frozen"; "" if huggingface_hub on PATH)
set -uo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
mira="$here/../mira"
RUN="${RUN:-pixi run --frozen}"

: "${HF_DATA_REPO:?set HF_DATA_REPO=youruser/racerx-mira-wds}"
RX_ROOT="${RX_ROOT:-/mnt/c/recordings/mira_wds}"
[ -f "$RX_ROOT/train/index.json" ] || { echo "ERROR: no data at $RX_ROOT/train -- build the WebDataset first"; exit 1; }

echo "Uploading $RX_ROOT/{train,test} -> HF dataset $HF_DATA_REPO (private) ..."
cd "$mira"
HF_DATA_REPO="$HF_DATA_REPO" RX_ROOT="$RX_ROOT" $RUN python - <<'PY' || { echo "UPLOAD FAILED"; exit 1; }
import os
from huggingface_hub import HfApi
repo = os.environ["HF_DATA_REPO"]
root = os.environ["RX_ROOT"]
api = HfApi()
api.create_repo(repo, repo_type="dataset", private=True, exist_ok=True)
for s in ("train", "test"):
    d = os.path.join(root, s)
    if os.path.isdir(d):
        api.upload_folder(folder_path=d, repo_id=repo, repo_type="dataset", path_in_repo=s)
        print("uploaded", s)
PY

echo
echo "Done. On the Horde instance:  HF_DATA_REPO=$HF_DATA_REPO ./train_horde.sh"
