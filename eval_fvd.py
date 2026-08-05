#!/usr/bin/env python3
"""Evaluate video quality using FVD + PSNR"""

import os
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["TORCH_COMPILE_DISABLE"] = "1"

import torch
import sys
import math
import argparse
from pathlib import Path
import numpy as np

try:
    from torchcodec.decoders import VideoDecoder
except ImportError as e:
    print(f"ERROR: torchcodec not found: {e}")
    sys.exit(1)

try:
    import lpips
    HAS_LPIPS = True
except ImportError:
    HAS_LPIPS = False

from skimage.metrics import structural_similarity as ssim

def load_video(video_path: str, num_frames: int = 32):
    """Load video frames"""
    path = Path(video_path)
    if not path.exists():
        print(f"ERROR: {path} not found")
        return None

    try:
        dec = VideoDecoder(str(path))
        frames = torch.stack([f.cpu() for f in dec])[:num_frames]
        # (T, C, H, W) -> (1, C, T, H, W) for I3D
        frames = frames.permute(1, 0, 2, 3).unsqueeze(0).float() / 255.0
        return frames
    except Exception as e:
        print(f"ERROR: Failed to load video: {e}")
        return None

def compute_psnr(real_video, gen_video):
    """Compute PSNR between real and generated video (frame-wise mean)"""
    # Videos in [0, 1] range
    # Handle different frame counts
    min_frames = min(real_video.shape[2], gen_video.shape[2])
    real_trimmed = real_video[:, :, :min_frames, :, :]
    gen_trimmed = gen_video[:, :, :min_frames, :, :]

    mse = torch.mean((real_trimmed - gen_trimmed) ** 2).item()
    if mse == 0:
        return 100.0
    psnr = -10.0 * math.log10(mse)
    return psnr

def eval_fvd(real_video, gen_video):
    """Compute FVD between real and generated video"""
    if not HAS_FVD:
        print("ERROR: fvd.py not found")
        return None

    print("Computing FVD...")

    try:
        # Videos shape: (1, C, T, H, W)
        # compute_fvd requires: real_videos, fake_videos, max_items, device, batch_size
        fvd_score = compute_fvd(
            real_video,
            gen_video,
            max_items=min(real_video.shape[2], gen_video.shape[2]),  # Use frame count
            device="cuda",
            batch_size=8
        )
        return float(fvd_score)
    except Exception as e:
        print(f"ERROR computing FVD: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    ap = argparse.ArgumentParser(description="Evaluate video quality using FVD")
    ap.add_argument("--real", default=r"C:\workspace\world\mira\dataset_00000\2026-05-08T20-37-10Z-5f298d_c00000.p0.mp4",
                    help="ground truth video")
    ap.add_argument("--gen", help="generated video (required)")
    ap.add_argument("--num-frames", type=int, default=32, help="frames to evaluate")
    args = ap.parse_args()

    print("="*70)
    print("Video Quality Evaluation (FVD)")
    print("="*70)

    if not args.gen:
        print("ERROR: --gen required (generated video path)")
        return False

    # Load real video
    print(f"\nLoading real video: {Path(args.real).name}")
    real = load_video(args.real, args.num_frames)
    if real is None:
        return False
    print(f"✓ Real video shape: {real.shape}")

    # Load generated video
    print(f"\nLoading generated video: {Path(args.gen).name}")
    gen = load_video(args.gen, args.num_frames)
    if gen is None:
        return False
    print(f"✓ Generated video shape: {gen.shape}")

    # Compute PSNR
    print("\nComputing PSNR...")
    psnr = compute_psnr(real, gen)

    # Compute FVD
    fvd = eval_fvd(real, gen)

    if psnr is not None or fvd is not None:
        print(f"\n{'='*70}")
        print("Results")
        print(f"{'='*70}")

        if psnr is not None:
            print(f"PSNR: {psnr:.2f} dB (higher = better)")

        if fvd is not None:
            print(f"FVD:  {fvd:.2f} (lower = better)")
            print(f"\nFVD Interpretation:")
            print(f"  < 50:  Excellent")
            print(f"  50-75: Good")
            print(f"  > 75:  Poor")

        return True
    return False

if __name__ == "__main__":
    sys.exit(0 if main() else 1)
