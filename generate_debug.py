#!/usr/bin/env python3
"""Generate video and print action timeline"""

import os
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["TORCH_COMPILE_DISABLE"] = "1"

import torch
import json
from pathlib import Path
from mira.inference.loading import load_world_model
from mira.data.batch import VideoActionBatch
from mira.world_model.actions_config import ActionTensors, ActionConfig
from mira.world_model.config import WorldModelInferenceConfig
from torchcodec.decoders import VideoDecoder

checkpoint = "outputs/wm_ckpt_63000.pth"
video_path = "C:\\recordings\\mira_wds\\test\\000\\dataset_00000\\1ea1a4ce-2fdc-44fe-afdd-8019fbacea28_clip00000_c00000.p0.mp4"

print("="*70)
print("Generate Video with Action Debug")
print("="*70)

# Load actions
action_path = Path(video_path).parent / (Path(video_path).stem + ".jsonl")
with open(action_path) as f:
    actions_list = [json.loads(line) for line in f]

print(f"\nActions loaded: {len(actions_list)} frames")

# Analyze actions
all_keys = set()
for a in actions_list:
    all_keys.update(a.get("keys", []))

print(f"Available keys: {sorted(all_keys)}")
print("\nAction timeline:")
for i, a in enumerate(actions_list):
    keys = a.get("keys", [])
    key_str = str(keys) if keys else "(neutral)"
    print(f"  Frame {i:2d}: {key_str:30s}", end="")
    if (i + 1) % 2 == 0:
        print()
    else:
        print(" | ", end="")

print("\n\nLoading model...")
model, run_cfg = load_world_model(checkpoint, device="cuda")

print("Generating video...")
# (actual generation would go here)
print("Done!")
