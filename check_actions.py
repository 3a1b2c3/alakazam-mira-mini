#!/usr/bin/env python3
"""Validate action file and show action timeline"""

import json
import sys
from pathlib import Path

def main():
    if len(sys.argv) < 2:
        print("Usage: check_actions.py <video_path>")
        sys.exit(1)

    video_path = Path(sys.argv[1])
    action_path = video_path.parent / (video_path.stem + ".jsonl")

    if not action_path.exists():
        print(f"ERROR: Action file not found: {action_path}")
        sys.exit(1)

    try:
        actions_list = []
        with open(action_path, 'r', encoding='utf-8', errors='replace') as f:
            for i, line in enumerate(f):
                if line.strip():
                    try:
                        actions_list.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        print(f"  Warning: Frame {i} parse error: {e}", file=sys.stderr)

        if not actions_list:
            print("ERROR: No valid actions in file")
            sys.exit(1)

        # Extract all unique keys
        all_keys = set()
        for a in actions_list:
            all_keys.update(a.get("keys", []))

        print(f"Actions: {len(actions_list)} frames")
        print(f"Keys: {sorted(all_keys) if all_keys else '(neutral only)'}")

        return 0

    except Exception as e:
        print(f"ERROR: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
