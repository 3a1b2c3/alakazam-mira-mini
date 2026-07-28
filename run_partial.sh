#!/usr/bin/env bash
# Build index.partial.json (present shards only) then finetune on them -- for an
# INCOMPLETE upload: drops entries whose .tar isn't on disk so there's no
# missing-shard FileNotFoundError. Re-run any time more shards land to widen the
# training set (auto-resumes from the last checkpoint). Once all shards are
# present, use finetune_racerx.sh directly instead.
#
#   RX_ROOT   mira_wds dir (holds train/ + test/); default /home/horde/mira_wds
#   $1        steps (default 10000); extra Hydra overrides pass through
# Usage:
#   /home/horde/alakazam-mira-mini/run_partial.sh
#   /home/horde/alakazam-mira-mini/run_partial.sh 20000 dataloader.num_workers=0
set -uo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
REC="${RX_ROOT:-/home/horde/mira_wds}"
[ -f "$REC/train/index.json" ] || { echo "ERROR: no train index at $REC/train/index.json"; exit 1; }

# $1 is steps only if all digits; a key=value first arg passes through as override.
STEPS="${1:-10000}"; case "$STEPS" in ''|*[!0-9]*) STEPS=10000;; *) shift;; esac

for split in train test; do
    [ -f "$REC/$split/index.json" ] || continue
    python3 - "$REC/$split" <<'PY'
import json, os, sys
root = sys.argv[1]
d = json.load(open(f"{root}/index.json"))
keep = [e for e in d["entries"] if os.path.exists(os.path.join(root, e["shard"]))]
d["entries"] = keep; d["total_samples"] = len(keep)
json.dump(d, open(f"{root}/index.partial.json", "w"))
print(f"{os.path.basename(root)}: kept {len(keep)} entries")
if not keep:
    sys.exit(f"ERROR: 0 shards present under {root}")
PY
    [ $? -eq 0 ] || exit 1
done

exec "$here/finetune_racerx.sh" "$STEPS" \
    dataset.train_index="$REC/train/index.partial.json" \
    dataset.test_index="$REC/test/index.partial.json" "$@"
