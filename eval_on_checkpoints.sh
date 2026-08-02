#!/usr/bin/env bash
# Watch a training run's output dir and run the offline PSNR/LPIPS/SSIM eval on
# EACH new checkpoint-<step>/checkpoint.pth as it lands, logging the scalars to a
# TB event dir so PSNR shows up as a curve. This is independent of the in-training
# downstream eval (which only fires every validation.downstream_val_every steps),
# so you get PSNR per checkpoint instead of waiting for step 15000.
#
# Prereqs: same as eval_wm.sh -- mira pixi env + DINO weights (download_dino.sh:
# the metrics need vitb16, else the eval self-skips and no PSNR is produced).
#
# Usage:
#   ./eval_on_checkpoints.sh <output_dir> [-- <extra eval_wm args>]
#     <output_dir>  the run's run.output_dir (e.g. train_world_model_logs_scratch
#                   or train_world_model_logs_finetune), relative to ../mira or absolute.
#   env: INTERVAL (poll seconds, default 60), SAMPLES (--num-samples, default 64)
#
# View it:  point tensorboard at <output_dir>/eval_tb (or the parent mira dir) and
#           look at eval/psnr, eval/lpips, eval/ssim.
set -uo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
mira="$here/../mira"
RUN="${RUN:-pixi run --frozen}"
INTERVAL="${INTERVAL:-60}"
SAMPLES="${SAMPLES:-64}"

[ $# -ge 1 ] || { echo "usage: $0 <output_dir> [-- <extra eval_wm args>]"; exit 1; }
OUT="$1"; shift
# strip a leading "--" separator if present, rest are extra eval_wm args
[ "${1:-}" = "--" ] && shift
EXTRA=("$@")

# Convenience: accept a checkpoint.pth or a checkpoint-<N> dir and resolve to the
# run's output dir (so passing the ckpt path works too -- it watches that whole run).
case "$OUT" in
    */checkpoint.pth)    OUT="$(dirname "$(dirname "$OUT")")";;
    */checkpoint-*[0-9]) OUT="$(dirname "$OUT")";;
esac

# resolve output_dir: absolute as-is, else relative to the mira repo (where runs live)
case "$OUT" in /*) OUTDIR="$OUT";; *) OUTDIR="$mira/$OUT";; esac
TBDIR="$OUTDIR/eval_tb"
LOG="$OUTDIR/eval_on_checkpoints.log"
DONE="$TBDIR/.evaluated_steps"
mkdir -p "$TBDIR"; touch "$DONE"

echo "watching $OUTDIR for new checkpoints (interval ${INTERVAL}s, samples ${SAMPLES})"
echo "  eval TB -> $TBDIR   (tag eval/psnr)"
echo "  log     -> $LOG"

write_tb() {  # step psnr lpips ssim  -> TB scalars
    ( cd "$mira" && $RUN python - "$TBDIR" "$1" "$2" "$3" "$4" <<'PY'
import sys
from torch.utils.tensorboard import SummaryWriter
tb, step = sys.argv[1], int(sys.argv[2])
w = SummaryWriter(tb)
for name, val in zip(("psnr", "lpips", "ssim"), sys.argv[3:6]):
    try:
        w.add_scalar(f"eval/{name}", float(val), step)
    except ValueError:
        pass
w.close()
PY
    )
}

while true; do
    # newest-first so if several appear at once we still do them all (sorted below)
    for ckpt in "$OUTDIR"/checkpoint-*/checkpoint.pth; do
        [ -f "$ckpt" ] || continue
        d="$(dirname "$ckpt")"; step="${d##*/checkpoint-}"
        case "$step" in *[!0-9]*) continue;; esac      # skip .checkpoint-N.tmp etc.
        grep -qx "$step" "$DONE" && continue           # already evaluated

        echo "=== [$(date '+%H:%M:%S')] eval checkpoint-$step ===" | tee -a "$LOG"
        out="$("$here/eval_wm.sh" "$ckpt" --skip-validation --num-samples "$SAMPLES" "${EXTRA[@]}" 2>&1)"
        printf '%s\n' "$out" >> "$LOG"
        # top-level metric lines are printed as "  psnr: NN.NNNN" (exclude action/*/psnr)
        val() { printf '%s\n' "$out" | grep -iE "^[[:space:]]*$1:[[:space:]]" | grep -oE "[0-9]+\.[0-9]+" | tail -1; }
        psnr="$(val psnr)"; lpips="$(val lpips)"; ssim="$(val ssim)"
        if [ -n "$psnr" ]; then
            echo "  step $step -> psnr=$psnr lpips=${lpips:-?} ssim=${ssim:-?}" | tee -a "$LOG"
            write_tb "$step" "$psnr" "${lpips:-nan}" "${ssim:-nan}"
        else
            echo "  step $step -> NO PSNR (metrics skipped? need vitb16 -> download_dino.sh)" | tee -a "$LOG"
        fi
        echo "$step" >> "$DONE"
    done
    sleep "$INTERVAL"
done
