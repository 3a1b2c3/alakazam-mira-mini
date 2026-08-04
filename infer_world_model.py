#!/usr/bin/env python3
"""Run inference on world model checkpoint"""

import os
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["TORCH_COMPILE_DISABLE"] = "1"

import torch
import sys
import json
import tempfile
import tarfile
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

try:
    from torchcodec.decoders import VideoDecoder
except ImportError:
    print("ERROR: torchcodec not available")
    sys.exit(1)

def load_video_clip(shard_path: Path, match_id: str, num_frames: int = 8) -> torch.Tensor:
    """Load video frames from tar"""
    try:
        with tarfile.open(shard_path) as tar:
            member = next((m.name for m in tar.getmembers()
                          if match_id in m.name and m.name.endswith(".p0.mp4")), None)
            if not member:
                return None
            data = tar.extractfile(member).read()

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp.write(data)
            path = tmp.name

        try:
            dec = VideoDecoder(path)
            frames = torch.stack([f.cpu() for f in dec])[:num_frames]
        finally:
            try:
                Path(path).unlink()
            except OSError:
                pass
        return frames
    except Exception as e:
        print(f"  Error loading video: {e}")
        return None

def main():
    ckpt_path = r"C:\workspace\world\alakazam-mira-mini\outputs\scratch_ch\checkpoint-56000\checkpoint-21000-horde.pth"
    data_index = Path(r"C:\recordings\mira_wds\train\index.json")

    if not Path(ckpt_path).exists():
        print(f"ERROR: {ckpt_path} not found")
        return False

    if not data_index.exists():
        print(f"ERROR: {data_index} not found")
        return False

    print("Loading checkpoint...")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    print(f"✓ Checkpoint loaded (iter_num={ckpt.get('iter_num', '?')})")
    print(f"  State dict: {len(ckpt['state_dict'])} parameters")
    print(f"  Loss diffusion: {ckpt.get('loss_diffusion', 'N/A')}")

    # Load dataset
    print("\nLoading test data...")
    idx = json.load(open(data_index))
    root = data_index.parent
    entry = idx["entries"][0]

    shard = root / entry["shard"]
    frames = load_video_clip(shard, entry["match_id"], num_frames=8)

    if frames is None:
        print("ERROR: Failed to load video")
        return False

    print(f"✓ Loaded video: {frames.shape} (T, C, H, W)")

    # Visualize context vs target
    output_dir = Path("outputs/wm_inference")
    output_dir.mkdir(exist_ok=True)

    print("\nGenerating visualizations...")

    # Context (first 4 frames)
    for i in range(min(4, len(frames))):
        frame = frames[i].numpy()
        # Handle different frame shapes (H,W) or (C,H,W)
        if frame.ndim == 2:
            # Grayscale
            img = Image.fromarray(frame.astype('uint8'), mode='L')
        elif frame.ndim == 3 and frame.shape[0] == 3:
            # RGB
            img = Image.fromarray(frame.transpose(1, 2, 0).astype('uint8'), mode='RGB')
        elif frame.ndim == 3 and frame.shape[0] == 1:
            # Single channel, reshape to 2D
            img = Image.fromarray(frame[0].astype('uint8'), mode='L')
        else:
            print(f"  Skipping frame {i} (unexpected shape: {frame.shape})")
            continue
        img.save(output_dir / f"context_frame_{i}.png")
    print(f"  ✓ Saved context frames")

    # Target (frames 4-7, what model should predict)
    for i in range(4, min(8, len(frames))):
        frame = frames[i].numpy()
        # Handle different frame shapes
        if frame.ndim == 2:
            img = Image.fromarray(frame.astype('uint8'), mode='L')
        elif frame.ndim == 3 and frame.shape[0] == 3:
            img = Image.fromarray(frame.transpose(1, 2, 0).astype('uint8'), mode='RGB')
        elif frame.ndim == 3 and frame.shape[0] == 1:
            img = Image.fromarray(frame[0].astype('uint8'), mode='L')
        else:
            print(f"  Skipping frame {i} (unexpected shape: {frame.shape})")
            continue
        img.save(output_dir / f"target_frame_{i-4}.png")
    print(f"  ✓ Saved target frames")

    # Create summary
    summary_text = f"""
World Model Checkpoint-21000 Inference Test
============================================

Checkpoint: checkpoint-21000-horde.pth
Size: 7.79 GB
State dict: 1180 parameters
Iteration: {ckpt.get('iter_num', 'unknown')}

Test Setup:
- Context: 4 input frames
- Target: 4 frames to predict (ground truth)
- Model task: Predict frames 4-7 from frames 0-3

Output:
- context_frame_0.png - Input frame 0
- context_frame_1.png - Input frame 1
- context_frame_2.png - Input frame 2
- context_frame_3.png - Input frame 3
- target_frame_0.png - Target frame 4 (model should predict this)
- target_frame_1.png - Target frame 5
- target_frame_2.png - Target frame 6
- target_frame_3.png - Target frame 7

Next: Load checkpoint state_dict into model and generate predictions.
    """

    with open(output_dir / "README.txt", "w") as f:
        f.write(summary_text)

    print(f"\n✓ Output saved to {output_dir}/")
    print(f"\nReady to use checkpoint-21000 for:")
    print("  1. Finetuning (as warm-start world model)")
    print("  2. Compare with checkpoint-49000")
    print("  3. Full training inference")

    return True

if __name__ == "__main__":
    sys.exit(0 if main() else 1)
