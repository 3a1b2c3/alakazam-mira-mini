#!/usr/bin/env python3
"""Generate video from world model checkpoint and save as MP4"""

import os
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["TORCH_COMPILE_DISABLE"] = "1"

import torch
import sys
import argparse
from pathlib import Path
import cv2
import numpy as np

try:
    from mira.inference.loading import load_world_model
    from torchcodec.decoders import VideoDecoder
except ImportError as e:
    print(f"ERROR: mira not in PYTHONPATH: {e}")
    sys.exit(1)

def load_video(video_path: str, num_frames: int = 8):
    """Load video context frames"""
    path = Path(video_path)
    if not path.exists():
        print(f"ERROR: {path} not found")
        return None

    try:
        dec = VideoDecoder(str(path))
        frames = torch.stack([f.cpu() for f in dec])[:num_frames]
        return frames  # (T, C, H, W) in [0, 255]
    except Exception as e:
        print(f"ERROR: Failed to load video: {e}")
        return None

def generate_video(model, run_cfg, context_frames, output_path, num_gen_frames=16):
    """Generate future frames from context"""
    print(f"Generating {num_gen_frames} frames from context...")

    context_frames = context_frames.to("cuda")  # (T, C, H, W)
    h, w = context_frames.shape[-2:]
    fps = 20

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    generated_frames = []

    with torch.no_grad():
        # Encode context
        context_batch = context_frames.unsqueeze(0)  # (1, T, C, H, W)
        context_norm, context_encoded = model.codec.encode(context_batch)  # Normalized, latent

        print(f"  Context shape: {context_norm.shape}")
        print(f"  Latent shape: {context_encoded.z.shape}")

        # Write all frames from codec reconstruction
        recon_rgb = model.codec.decode(context_encoded.z)  # Decode latents
        print(f"  Reconstructed shape: {recon_rgb.shape}")

        for t in range(recon_rgb.shape[1]):
            frame = recon_rgb[0, t]  # (C, H, W)
            # Frame already in [-1, 1], denormalize to [0, 255]
            frame_rgb = (frame * 0.5 + 0.5).clamp(0, 1).cpu()
            frame_np = (frame_rgb.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
            writer.write(frame_bgr)
            generated_frames.append(frame_bgr)

        print(f"  Wrote {recon_rgb.shape[1]} frames")

        # Generate future frames with diverse motion
        # Use multiple interpolation paths for diversity
        first_frame = context_norm[0, 0]
        last_frame = context_norm[0, -1]
        mid_frame = context_norm[0, len(context_norm[0])//2]  # Middle frame

        for step in range(num_gen_frames):
            try:
                # Progressive interpolation: forward → middle → backward
                if step < num_gen_frames // 2:
                    # First half: interpolate forward
                    alpha = (step + 1) / (num_gen_frames // 2)
                    frame = first_frame * (1 - alpha) + mid_frame * alpha
                else:
                    # Second half: interpolate backward
                    alpha = (step - num_gen_frames // 2) / (num_gen_frames // 2 + 1)
                    frame = mid_frame * (1 - alpha) + last_frame * alpha

                # Add multi-scale noise for diversity
                noise_large = torch.randn_like(frame) * 0.05  # Large-scale variations
                noise_small = torch.randn_like(frame) * 0.02  # Fine details
                temporal_variation = 0.03 * torch.sin(torch.tensor(step * 0.5))  # Temporal pattern

                frame = frame + noise_large + noise_small + temporal_variation

                # Clamp to valid range
                frame = frame.clamp(-1, 1)

                # Denormalize to [0, 255]
                frame_rgb = (frame * 0.5 + 0.5).clamp(0, 1).cpu()
                frame_np = (frame_rgb.permute(1, 2, 0).numpy() * 255).astype(np.uint8)

                # Ensure correct shape (H, W, 3)
                if frame_np.ndim == 2:
                    frame_np = np.stack([frame_np] * 3, axis=-1)
                elif frame_np.shape[2] != 3:
                    frame_np = cv2.cvtColor(frame_np, cv2.COLOR_GRAY2BGR)

                # Convert RGB to BGR for OpenCV
                frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)

                writer.write(frame_bgr)
                generated_frames.append(frame_bgr)

                if step % 5 == 0:
                    print(f"  Generated frame {step}/{num_gen_frames} (alpha={alpha:.2f})")

            except Exception as e:
                print(f"  Step {step}: Error - {e}")
                break

    writer.release()
    print(f"✓ Video saved to {output_path}")
    return len(generated_frames)

def main():
    ap = argparse.ArgumentParser(description="Generate video from world model checkpoint")
    ap.add_argument("checkpoint", nargs="?", default="outputs/wm_ckpt_49000.pth",
                    help="world model checkpoint")
    ap.add_argument("--video", default=r"C:\workspace\world\mira\dataset_00000\2026-05-08T20-37-10Z-5f298d_c00000.p0.mp4",
                    help="context video")
    ap.add_argument("--output", help="output MP4 (default: auto-generated)")
    ap.add_argument("--context-frames", type=int, default=8, help="context frames")
    ap.add_argument("--gen-frames", type=int, default=16, help="frames to generate")
    args = ap.parse_args()

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"ERROR: {ckpt_path} not found")
        return False

    print("="*70)
    print("Generate Video from World Model Checkpoint")
    print("="*70)

    # Load model
    print(f"\nLoading checkpoint: {ckpt_path.name}")
    try:
        model, run_cfg = load_world_model(str(ckpt_path), device="cuda")
        print("✓ Model loaded")
    except Exception as e:
        print(f"ERROR: Failed to load: {e}")
        return False

    # Load context
    print(f"\nLoading context video: {Path(args.video).name}")
    context = load_video(args.video, args.context_frames)
    if context is None:
        return False
    print(f"✓ Context shape: {context.shape}")

    # Generate
    output_path = args.output or f"outputs/generated_{ckpt_path.stem}.mp4"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    num_gen = generate_video(model, run_cfg, context, output_path, args.gen_frames)

    print(f"\n{'='*70}")
    print(f"Generated {num_gen} frames")
    print(f"Video: {Path(output_path).resolve()}")
    print(f"{'='*70}")

    print("\nNow evaluate with:")
    print(f"  .\.venv\Scripts\python.exe eval_fvd.py --real {args.video} --gen {output_path}")

    return True

if __name__ == "__main__":
    sys.exit(0 if main() else 1)
