#!/usr/bin/env bash
# Verify the trainer is actually WRITING TensorBoard scalars (not just that a dir
# exists). Finds the newest *tfevents* under <logdir>, then reads it and prints
# the scalar tags + latest step/value per tag. Confirms TB writes when expected.
#
# Interpreting the result:
#   scalar tags listed        -> TB is writing; steps should match the cadence
#                                (train: 0-9 every step, then every log_every=1%;
#                                 test/: step 0 then every val_every=2%)
#   "NO tfevents"             -> not writing yet. Check, in order:
#                                 1) run launched with ++tensorboard.logdir=<dir>
#                                 2) tensorboard importable in the PIXI env
#                                    (cd ../mira && pixi run --frozen python -c "import tensorboard")
#                                 3) it has passed the first val_first / early step
#   "file exists, NO scalars" -> writer created but no tb_log fired yet; wait a step
#
#   LOGDIR default ~/mira_wds/wm_racerx_ft/tb ; pass a dir to override
# Usage:
#   ./verify_tb.sh
#   ./verify_tb.sh /home/horde/mira_wds/wm_racerx_ft/tb
set -uo pipefail
LOGDIR="${1:-$HOME/mira_wds/wm_racerx_ft/tb}"
[ -d "$LOGDIR" ] || { echo "NO logdir: $LOGDIR"; exit 1; }

f="$(find "$LOGDIR" -name '*tfevents*' -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-)"
[ -n "$f" ] || { echo "NO tfevents under $LOGDIR -- TB not writing yet (see checklist in this script's header)"; exit 1; }
echo "newest event file: $f"
echo "size: $(stat -c %s "$f" 2>/dev/null) bytes, mtime: $(stat -c %y "$f" 2>/dev/null)"
echo

python3 - "$f" <<'PY'
import sys
try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
except ImportError:
    sys.exit("tensorboard not importable by this python3 -- pip install --user tensorboard")
ea = EventAccumulator(sys.argv[1]); ea.Reload()
tags = ea.Tags().get("scalars", [])
if not tags:
    print("file exists but has NO scalars yet -- SummaryWriter created, waiting for first tb_log")
    sys.exit(2)
print(f"{len(tags)} scalar tags -- latest step/value per tag:")
for t in sorted(tags):
    s = ea.Scalars(t)
    print(f"  {t:40s} step={s[-1].step:<7} value={s[-1].value:.5f}  ({len(s)} points, first step={s[0].step})")
PY
