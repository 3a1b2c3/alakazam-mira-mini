#!/usr/bin/env python3
"""Evaluate video quality: PSNR + SSIM + LPIPS"""

import os
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["TORCH_COMPILE_DISABLE"] = "1"

import torch
import sys
import math
import argparse
from pathlib import Path
import numpy as np
from skimage.metrics import structural_similarity as ssim
from torch.utils.tensorboard import SummaryWriter

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

def load_video(video_path: str, num_frames: int = 32):
    """Load video frames"""
    path = Path(video_path)
    if not path.exists():
        print(f"ERROR: {path} not found")
        return None

    try:
        dec = VideoDecoder(str(path))
        frames = torch.stack([f.cpu() for f in dec])[:num_frames]
        # (T, C, H, W) in [0, 255]
        return frames
    except Exception as e:
        print(f"ERROR: Failed to load video: {e}")
        return None

def compute_psnr(real_video, gen_video):
    """Compute mean PSNR"""
    import cv2
    min_frames = min(real_video.shape[0], gen_video.shape[0])
    real_trim = real_video[:min_frames].float() / 255.0
    gen_trim = gen_video[:min_frames].float() / 255.0

    # Handle resolution mismatch by downscaling real_video to gen_video resolution
    if real_trim.shape[1:] != gen_trim.shape[1:]:
        print(f"  Resolution mismatch: {real_trim.shape[1:]} vs {gen_trim.shape[1:]}")
        print(f"  Downscaling real video to match generated...")
        gen_h, gen_w = gen_trim.shape[2:4]  # (T, C, H, W) -> H, W
        real_resized = []
        for i in range(real_trim.shape[0]):
            frame_np = (real_trim[i].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            frame_resized = cv2.resize(frame_np, (gen_w, gen_h), interpolation=cv2.INTER_LANCZOS4)
            frame_tensor = torch.from_numpy(frame_resized).permute(2, 0, 1).float() / 255.0  # (C, H, W)
            real_resized.append(frame_tensor)
        real_trim = torch.stack(real_resized)

    mse = torch.mean((real_trim - gen_trim) ** 2).item()
    if mse == 0:
        return 100.0
    psnr = -10.0 * math.log10(mse)
    return psnr

def compute_ssim(real_video, gen_video):
    """Compute mean SSIM"""
    import cv2
    min_frames = min(real_video.shape[0], gen_video.shape[0])
    real_trim = real_video[:min_frames].float() / 255.0
    gen_trim = gen_video[:min_frames].float() / 255.0

    # Handle resolution mismatch
    if real_trim.shape[1:] != gen_trim.shape[1:]:
        gen_h, gen_w = gen_trim.shape[2:4]  # (T, C, H, W) -> H, W
        real_resized = []
        for i in range(real_trim.shape[0]):
            frame_np = (real_trim[i].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            frame_resized = cv2.resize(frame_np, (gen_w, gen_h), interpolation=cv2.INTER_LANCZOS4)
            frame_tensor = torch.from_numpy(frame_resized).permute(2, 0, 1).float() / 255.0  # (C, H, W)
            real_resized.append(frame_tensor)
        real_trim = torch.stack(real_resized)

    ssim_scores = []
    for i in range(min_frames):
        # Convert to numpy (H, W, C)
        real_frame = real_trim[i].permute(1, 2, 0).numpy()
        gen_frame = gen_trim[i].permute(1, 2, 0).numpy()

        # Compute SSIM
        score = ssim(real_frame, gen_frame, channel_axis=2, data_range=1.0)
        ssim_scores.append(score)

    return np.mean(ssim_scores)

def compute_lpips(real_video, gen_video):
    """Compute mean LPIPS"""
    if not HAS_LPIPS:
        return None

    min_frames = min(real_video.shape[0], gen_video.shape[0])
    real_trim = real_video[:min_frames].float() / 255.0
    gen_trim = gen_video[:min_frames].float() / 255.0

    # Normalize to [-1, 1]
    real_trim = real_trim * 2 - 1
    gen_trim = gen_trim * 2 - 1

    try:
        loss_fn = lpips.LPIPS(net='alex')
        loss_fn.to("cuda")

        lpips_scores = []
        for i in range(min_frames):
            real_frame = real_trim[i].unsqueeze(0).to("cuda")
            gen_frame = gen_trim[i].unsqueeze(0).to("cuda")

            with torch.no_grad():
                score = loss_fn(real_frame, gen_frame).item()
            lpips_scores.append(score)

        return np.mean(lpips_scores)
    except Exception as e:
        print(f"  LPIPS error: {e}")
        return None

def main():
    ap = argparse.ArgumentParser(description="Evaluate video quality: PSNR + SSIM + LPIPS")
    ap.add_argument("--real", default=r"C:\workspace\world\mira\dataset_00000\2026-05-08T20-37-10Z-5f298d_c00000.p0.mp4",
                    help="ground truth video")
    ap.add_argument("--gen", help="generated video (required)")
    ap.add_argument("--num-frames", type=int, default=32, help="frames to evaluate")
    ap.add_argument("--tb-dir", default="outputs/tb", help="TensorBoard log directory")
    ap.add_argument("--step", type=int, default=0, help="training step for logging")
    args = ap.parse_args()

    # Setup TensorBoard
    writer = SummaryWriter(args.tb_dir)

    print("="*70)
    print("Video Quality Evaluation")
    print("="*70)

    if not args.gen:
        print("ERROR: --gen required")
        return False

    # Load videos
    print(f"\nLoading real video: {Path(args.real).name}")
    real = load_video(args.real, args.num_frames)
    if real is None:
        return False
    print(f"✓ Real video: {real.shape}")

    print(f"\nLoading generated video: {Path(args.gen).name}")
    gen = load_video(args.gen, args.num_frames)
    if gen is None:
        return False
    print(f"✓ Generated video: {gen.shape}")

    # Compute metrics
    print("\n" + "="*70)
    print("Computing metrics...")
    print("="*70)

    psnr = compute_psnr(real, gen)
    print(f"PSNR: {psnr:.2f} dB (higher = better)")

    print("Computing SSIM...")
    ssim_score = compute_ssim(real, gen)
    print(f"SSIM: {ssim_score:.4f} (higher = better, range 0-1)")

    if HAS_LPIPS:
        print("Computing LPIPS...")
        lpips_score = compute_lpips(real, gen)
        if lpips_score is not None:
            print(f"LPIPS: {lpips_score:.4f} (lower = better)")
    else:
        print("LPIPS: not available (install: pip install lpips)")
        lpips_score = None

    # Summary
    print("\n" + "="*70)
    print("Summary")
    print("="*70)
    print(f"PSNR:  {psnr:.2f} dB")
    print(f"SSIM:  {ssim_score:.4f}")
    if lpips_score is not None:
        print(f"LPIPS: {lpips_score:.4f}")

    # Log to TensorBoard
    print("\nLogging to TensorBoard...")
    writer.add_scalar("eval/psnr", psnr, args.step)
    writer.add_scalar("eval/ssim", ssim_score, args.step)
    if lpips_score is not None:
        writer.add_scalar("eval/lpips", lpips_score, args.step)
    writer.flush()
    writer.close()

    print(f"✓ Logged to {args.tb_dir}")
    return True

if __name__ == "__main__":
    sys.exit(0 if main() else 1)
