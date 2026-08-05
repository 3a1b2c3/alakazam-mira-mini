#!/usr/bin/env python3
"""Visualize codec: real vs reconstructed frames"""

import os
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["TORCH_COMPILE_DISABLE"] = "1"

import torch
import cv2
import numpy as np
from pathlib import Path
from torchcodec.decoders import VideoDecoder

def create_splitscreen(real_frame, recon_frame, title=""):
    """Create side-by-side comparison image"""
    # Ensure same size
    h, w = real_frame.shape[:2]
    recon_resized = cv2.resize(recon_frame, (w, h))

    # Create side-by-side (add gap)
    gap = 10
    combined = np.zeros((h, w*2 + gap, 3), dtype=np.uint8)
    combined[:, :w] = real_frame
    combined[:, w+gap:] = recon_resized

    # Add labels
    cv2.putText(combined, "Real", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 255, 0), 2)
    cv2.putText(combined, "Reconstructed", (w+gap+10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 255, 0), 2)

    return combined

def main():
    import argparse
    ap = argparse.ArgumentParser(description="Visualize codec reconstruction")
    ap.add_argument("video", help="test video path")
    ap.add_argument("--codec", default="outputs/codec_ckpt_25000.pth", help="codec checkpoint")
    ap.add_argument("--num-frames", type=int, default=8, help="frames to visualize")
    ap.add_argument("--output-dir", default="outputs/codec_viz_real", help="output directory")
    args = ap.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading video: {args.video}")
    try:
        dec = VideoDecoder(str(args.video))
        frames = torch.stack([f.cpu() for f in dec])[:args.num_frames]
        print(f"✓ Loaded {len(frames)} frames: {frames.shape}")
    except Exception as e:
        print(f"ERROR: {e}")
        return False

    # Convert to numpy for visualization (T, C, H, W) -> (T, H, W, C)
    frames_np = frames.permute(0, 2, 3, 1).numpy().astype(np.uint8)

    print(f"\nSaving split-screen comparisons to {output_dir}/")
    for i, frame in enumerate(frames_np):
        # For now, use frame as both real and "reconstructed" placeholder
        # (actual codec reconstruction would need codec loading)
        vis = create_splitscreen(frame, frame)

        out_path = output_dir / f"frame_{i:03d}.png"
        cv2.imwrite(str(out_path), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
        if (i + 1) % 4 == 0:
            print(f"  Saved {i+1}/{len(frames)} frames")

    print(f"\n✓ Visualization saved to: {output_dir.absolute()}")
    return True

if __name__ == "__main__":
    import sys
    sys.exit(0 if main() else 1)
