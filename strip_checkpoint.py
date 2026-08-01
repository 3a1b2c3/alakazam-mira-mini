#!/usr/bin/env python3
"""Strip optimizer state from checkpoint to reduce size for inference/archival.
Reduces 29 GB checkpoint → ~5-10 GB weights-only checkpoint.
Usage: python strip_checkpoint.py <checkpoint.pth> [--output <out.pth>]
"""
import torch
import argparse
from pathlib import Path

def strip_checkpoint(ckpt_path, output_path=None):
    """Load checkpoint and save only model weights."""
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    print(f"Loading {ckpt_path.name}...")
    ckpt = torch.load(ckpt_path, map_location="cpu")

    original_size = ckpt_path.stat().st_size / 1e9
    print(f"  Original size: {original_size:.1f} GB")

    # Extract model weights only
    if "model" in ckpt:
        weights_only = {"model": ckpt["model"]}
    else:
        # Fallback: assume entire checkpoint is model state
        weights_only = ckpt

    # Determine output path
    if output_path is None:
        output_path = ckpt_path.parent / f"{ckpt_path.stem}-weights-only.pth"
    else:
        output_path = Path(output_path)

    print(f"Saving weights to {output_path.name}...")
    torch.save(weights_only, output_path)

    new_size = output_path.stat().st_size / 1e9
    reduction = (1 - new_size / original_size) * 100
    print(f"  Stripped size: {new_size:.1f} GB ({reduction:.0f}% smaller)")
    print(f"✓ Complete: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Strip optimizer state from MIRA checkpoint")
    parser.add_argument("checkpoint", help="Path to checkpoint.pth")
    parser.add_argument("--output", "-o", help="Output path (default: {name}-weights-only.pth)")
    args = parser.parse_args()

    try:
        strip_checkpoint(args.checkpoint, args.output)
    except Exception as e:
        print(f"ERROR: {e}")
        exit(1)
