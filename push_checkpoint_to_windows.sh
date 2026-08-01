#!/usr/bin/env bash
# Push most recent MIRA checkpoint from Horde to Windows machine.
# Setup: Configure ~/.ssh/config with Windows host, or edit WINDOWS_HOST below.
# Usage: ./push_checkpoint_to_windows.sh

set -uo pipefail
WINDOWS_HOST="windows-machine"
HORDE_PATH="$HOME/mira_wds/wm_racerx_ft"
WINDOWS_PATH="C:\\workspace\\world\\mira-checkpoints"

# Get latest checkpoint
latest=$(ls -d "$HORDE_PATH"/checkpoint-* 2>/dev/null | sed 's/.*checkpoint-//' | sort -n | tail -1)
if [ -z "$latest" ]; then
    echo "ERROR: No checkpoints found at $HORDE_PATH"
    exit 1
fi

echo "Pushing checkpoint-$latest to $WINDOWS_HOST..."
rsync -avz --progress "$HORDE_PATH/checkpoint-$latest/" "$WINDOWS_HOST:$WINDOWS_PATH/checkpoint-$latest/" || \
scp -r "$HORDE_PATH/checkpoint-$latest/checkpoint.pth" "$WINDOWS_HOST:$WINDOWS_PATH/checkpoint-$latest/"

if [ $? -eq 0 ]; then
    echo "✓ Complete: checkpoint-$latest on $WINDOWS_HOST"
else
    echo "✗ Failed"
    exit 1
fi
