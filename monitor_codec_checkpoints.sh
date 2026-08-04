#!/bin/bash
# Monitor codec training and run PSNR eval on each checkpoint
# Usage: bash monitor_codec_checkpoints.sh [codec_logs_dir] [test_data_dir] [num_samples]
# Run in background while train_codec.sh is active

set -uo pipefail

CODEC_LOGS="${1:-codec_logs}"
TEST_DATA="${2:-.}"
NUM_SAMPLES="${3:-16}"
PSNR_SCRIPT="${4:-codec_recon_psnr.py}"

echo "============================================================"
echo "Codec Checkpoint Monitor"
echo "============================================================"
echo "Monitoring: $CODEC_LOGS"
echo "Test data: $TEST_DATA"
echo "PSNR script: $PSNR_SCRIPT"
echo ""

# Track evaluated checkpoints
declare -A evaluated

while true; do
    # Find latest checkpoint
    latest=$(ls -d "$CODEC_LOGS"/checkpoint-*/checkpoint.pth 2>/dev/null | \
        sed 's#.*/checkpoint-\([0-9]*\)/.*#\1#' | \
        sort -n | tail -1)

    if [ -z "$latest" ]; then
        echo "[$(date '+%H:%M:%S')] No checkpoints yet, waiting..."
        sleep 10
        continue
    fi

    # Check if already evaluated
    if [ -n "${evaluated[$latest]:-}" ]; then
        sleep 5
        continue
    fi

    # Run PSNR evaluation
    ckpt_path="$CODEC_LOGS/checkpoint-$latest/checkpoint.pth"
    output_file="psnr_ckpt_${latest}.txt"

    echo "[$(date '+%H:%M:%S')] Evaluating checkpoint-$latest..."

    python "$PSNR_SCRIPT" \
        --checkpoint "$ckpt_path" \
        --num-samples "$NUM_SAMPLES" \
        --test-root "$TEST_DATA" \
        2>&1 | tee "$output_file"

    # Extract PSNR from output
    psnr=$(grep "^PSNR:" "$output_file" | tail -1 | awk '{print $2}')

    if [ -n "$psnr" ]; then
        echo "[$(date '+%H:%M:%S')] ✓ checkpoint-$latest PSNR: $psnr dB" | tee -a codec_psnr_log.txt
    else
        echo "[$(date '+%H:%M:%S')] ✗ checkpoint-$latest PSNR eval failed"
    fi

    evaluated[$latest]=1
    sleep 10
done
