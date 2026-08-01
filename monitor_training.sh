#!/usr/bin/env bash
# Monitor MIRA finetune training on Horde (run this ON Horde).
# Usage: ./monitor_training.sh [log|gpu|all|follow]
#   log   - show last 50 log lines (default)
#   gpu   - show GPU usage
#   all   - show both log + GPU
#   follow - stream live log

set -uo pipefail
LOG="$HOME/alakazam-mira-mini/ft.log"
MODE="${1:-log}"

show_log() {
    echo "=== Training Log (last 50 lines) ==="
    tail -50 "$LOG" 2>/dev/null || echo "Log not found: $LOG"
}

show_gpu() {
    echo "=== GPU Status ==="
    nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,utilization.memory --format=csv,noheader
}

show_checkpoint() {
    echo "=== Latest Checkpoint ==="
    latest=$(ls -d "$HOME/mira_wds/wm_racerx_ft"/checkpoint-* 2>/dev/null | sed 's/.*checkpoint-//' | sort -n | tail -1)
    if [ -n "$latest" ]; then
        echo "Checkpoint: checkpoint-$latest"
        size=$(du -sh "$HOME/mira_wds/wm_racerx_ft/checkpoint-$latest" 2>/dev/null | cut -f1)
        echo "Size: $size"
        mtime=$(stat -c %y "$HOME/mira_wds/wm_racerx_ft/checkpoint-$latest/checkpoint.pth" 2>/dev/null | cut -d' ' -f1-2)
        echo "Updated: $mtime"
    else
        echo "No checkpoints yet"
    fi
}

case "$MODE" in
    log)
        show_log
        echo
        show_checkpoint
        ;;
    gpu)
        show_gpu
        ;;
    all)
        show_log
        echo
        show_gpu
        echo
        show_checkpoint
        ;;
    follow)
        echo "Streaming live log (Ctrl+C to stop)..."
        tail -f "$LOG"
        ;;
    *)
        echo "Usage: $0 [log|gpu|all|follow]"
        exit 1
        ;;
esac
