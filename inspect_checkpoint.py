#!/usr/bin/env python3
"""Inspect checkpoint structure and info"""

import torch
import sys
from pathlib import Path

def inspect_checkpoint(ckpt_path):
    """Load and inspect checkpoint"""
    ckpt_path = Path(ckpt_path)

    if not ckpt_path.exists():
        print(f"ERROR: {ckpt_path} not found")
        return False

    print(f"Loading {ckpt_path.name}...")
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except Exception as e:
        print(f"ERROR: {e}")
        return False

    print("\n" + "="*60)
    print("Checkpoint Contents")
    print("="*60)

    # Show top-level keys
    if isinstance(ckpt, dict):
        print(f"Type: dict with {len(ckpt)} top-level keys")
        for key in list(ckpt.keys())[:10]:
            val = ckpt[key]
            if isinstance(val, torch.Tensor):
                print(f"  {key}: tensor {val.shape} {val.dtype}")
            elif isinstance(val, dict):
                print(f"  {key}: dict with {len(val)} items")
            else:
                print(f"  {key}: {type(val).__name__}")
    else:
        print(f"Type: {type(ckpt).__name__}")

    # Check for model state
    if isinstance(ckpt, dict):
        if "model" in ckpt:
            print(f"\n✓ Contains 'model' state")
            if isinstance(ckpt["model"], dict):
                print(f"  Model has {len(ckpt['model'])} parameters")
        if "state_dict" in ckpt:
            print(f"\n✓ Contains 'state_dict'")
            if isinstance(ckpt["state_dict"], dict):
                print(f"  State dict has {len(ckpt['state_dict'])} entries")
        if "step" in ckpt or "global_step" in ckpt:
            step_key = "step" if "step" in ckpt else "global_step"
            print(f"\n✓ Training step: {ckpt[step_key]}")

    print("\n" + "="*60)
    print(f"Checkpoint size: {ckpt_path.stat().st_size / (1024**3):.2f} GB")
    print("="*60)

    return True

if __name__ == "__main__":
    ckpt = r"C:\workspace\world\alakazam-mira-mini\outputs\scratch_ch\checkpoint-56000\checkpoint-21000-horde.pth"
    sys.exit(0 if inspect_checkpoint(ckpt) else 1)
