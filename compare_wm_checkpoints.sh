#!/usr/bin/env bash
# Compare world-model checkpoints (70k vs 98k)
# Uses eval_wm.sh to compute metrics on each checkpoint

set -uo pipefail
here="$(cd "$(dirname "$0")" && pwd)"

echo "============================================================"
echo "  World-Model Checkpoint Comparison"
echo "============================================================"
echo

# Two checkpoints to compare
CKPT_70K="$here/outputs/wm_ckpt_70000.pth"
CKPT_98K="$here/outputs/wm_ckpt_98000.pth"

# Verify they exist
if [ ! -f "$CKPT_70K" ]; then
    echo "ERROR: $CKPT_70K not found"
    exit 1
fi

if [ ! -f "$CKPT_98K" ]; then
    echo "ERROR: $CKPT_98K not found"
    exit 1
fi

echo "[1/2] Evaluating wm_ckpt_70000..."
echo
"$here/eval_wm.sh" "$CKPT_70K" --num-samples 16 --val-n-samples 8 | tee /tmp/eval_70k.txt
echo

echo
echo "[2/2] Evaluating wm_ckpt_98000..."
echo
"$here/eval_wm.sh" "$CKPT_98K" --num-samples 16 --val-n-samples 8 | tee /tmp/eval_98k.txt
echo

echo "============================================================"
echo "  Results Summary"
echo "============================================================"
echo
echo "[wm_ckpt_70000]:"
grep -E "val_loss|gFDD|gFID|PSNR|latent" /tmp/eval_70k.txt | head -10
echo
echo "[wm_ckpt_98000]:"
grep -E "val_loss|gFDD|gFID|PSNR|latent" /tmp/eval_98k.txt | head -10
echo

echo "Full output saved to:"
echo "  /tmp/eval_70k.txt"
echo "  /tmp/eval_98k.txt"
