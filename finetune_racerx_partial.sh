#!/usr/bin/env bash
# Finetune on ONLY the shards currently present under $RX_ROOT -- for when the
# upload is incomplete (some train/NNN/dataset_*.tar missing). Builds an
# index.partial.json per split that drops entries whose .tar isn't on disk, then
# hands off to finetune_racerx.sh with those partial indices (so no missing-shard
# FileNotFoundError). Once all shards land, run finetune_racerx.sh directly instead.
#
#   RX_ROOT   mira_wds dir (holds train/ + test/); default /home/horde/mira_wds
#   $1        steps (default 10000); extra Hydra overrides pass through
# Usage:
#   ./finetune_racerx_partial.sh                 10000 steps on present shards
#   ./finetune_racerx_partial.sh 20000
#   RX_ROOT=/data/mira_wds ./finetune_racerx_partial.sh 10000 dataloader.num_workers=0
set -uo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
REC="${RX_ROOT:-/home/horde/mira_wds}"
[ -f "$REC/train/index.json" ] || { echo "ERROR: no train index at $REC/train/index.json"; exit 1; }

# $1 is steps only if it's all digits; a key=value first arg passes through.
STEPS=10000
case "${1:-}" in
    ''|*[!0-9]*) : ;;
    *) STEPS="$1"; shift ;;
esac

# ---- build index.partial.json per split (present shards only) ----------------
OVERRIDES=()
for split in train test; do
    idx="$REC/$split/index.json"
    [ -f "$idx" ] || continue
    part="$REC/$split/index.partial.json"
    python3 - "$idx" "$REC/$split" "$part" <<'PY'
import json, os, sys
idx, root, out = sys.argv[1], sys.argv[2], sys.argv[3]
d = json.load(open(idx))
keep = [e for e in d["entries"] if os.path.exists(os.path.join(root, e["shard"]))]
d["entries"] = keep; d["total_samples"] = len(keep)
json.dump(d, open(out, "w"))
present = sorted({e["shard"] for e in keep})
print(f"{os.path.basename(os.path.dirname(idx))}: kept {len(keep)}/{len(d.get('entries', keep))} "
      f"entries, {len(present)} shards present")
if not keep:
    sys.exit(f"ERROR: 0 entries survive for {idx} -- no shards found under {root}")
PY
    [ $? -eq 0 ] || exit 1
    OVERRIDES+=("dataset.${split}_index=$part")
done

echo "== partial-index finetune ($STEPS steps) =="
printf '   %s\n' "${OVERRIDES[@]}"
exec "$here/finetune_racerx.sh" "$STEPS" "${OVERRIDES[@]}" "$@"
