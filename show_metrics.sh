#!/usr/bin/env bash
# Print the latest KEY training metrics from a run's TensorBoard logs -- no browser.
# Shows test/loss_total (the honest signal) + the world-model metrics (PSNR / DINO-Frechet /
# drift) once they start logging at downstream_val_every. Reads the tfevents directly via
# tensorboard's EventAccumulator (already in the pixi env).
#
#   RUN   env prefix (default "pixi run --frozen"; "" if the env is already active)
# Usage:
#   ./show_metrics.sh                      # newest run under ~/mira/* and ~/mira_wds/*
#   ./show_metrics.sh <run-dir | tb-dir>   # a specific run (e.g. .../wm_racerx_ft) or its tb/
#   watch -n 30 ./show_metrics.sh          # live-refresh every 30s
set -uo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
mira="$here/../mira"
RUN="${RUN:-pixi run --frozen}"
MIRA_WDS="${MIRA_WDS:-/home/horde/mira_wds}"

# Resolve the tb dir: explicit arg (run dir or tb dir), else the most-recently-written */tb.
TB="${1:-}"
if [ -n "$TB" ]; then
    [ -d "$TB/tb" ] && TB="$TB/tb"          # a run dir was passed -> use its tb/
else
    TB="$(ls -dt "$mira"/*/tb "$MIRA_WDS"/*/tb 2>/dev/null | head -1)"
fi
[ -n "$TB" ] && [ -d "$TB" ] || {
    echo "ERROR: no tb dir found. Pass one, e.g.:"
    echo "  ./show_metrics.sh $mira/train_world_model_logs_scratch"
    echo "  ./show_metrics.sh $MIRA_WDS/wm_racerx_ft"
    exit 1
}
echo "reading: $TB"
echo

cd "$mira"
exec $RUN python - "$TB" <<'PY'
import sys
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

tb = sys.argv[1]
ea = EventAccumulator(tb, size_guidance={"scalars": 0})
ea.Reload()
tags = ea.Tags().get("scalars", [])

def latest(tag):
    ev = ea.Scalars(tag)
    return (ev[-1].step, ev[-1].value) if ev else None

# Priority order: honest loss first, then the world-model quality metrics.
want = [
    "test/loss_total", "test/loss_diffusion", "learning_rate",
    "metrics/psnr", "metrics/ssim", "metrics/lpips",
    "metrics/dino_frechet_vs_recon", "metrics/dino_frechet_codec_floor",
    "metrics/dino_cos_drift", "metrics/dino_l2_drift", "metrics/latent_drift",
]
print(f"{'metric':<36}{'step':>9}{'value':>13}")
print("-" * 58)
shown = set()
for w in want:
    for tag in tags:
        if w in tag and tag not in shown:
            r = latest(tag)
            if r:
                print(f"{tag:<36}{r[0]:>9}{r[1]:>13.4f}")
            shown.add(tag)
# Any other metrics/* not in the priority list.
for tag in sorted(tags):
    if tag.startswith("metrics/") and tag not in shown:
        r = latest(tag)
        if r:
            print(f"{tag:<36}{r[0]:>9}{r[1]:>13.4f}")

if not any(t.startswith("metrics/") for t in tags):
    print()
    print("(no metrics/* yet -- the downstream eval hasn't run.")
    print(" First one lands at validation.downstream_val_every, e.g. step 15000.)")
PY
