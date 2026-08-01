#!/usr/bin/env python3
"""3D visualization of MIRA WebDataset samples.
Shows video frames + actions in interactive 3D space.
Usage: python visualize_sample.py --data <index.json> [--sample 0] [--port 8080]
"""
import json
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import torch
from torchcodec.decoders import VideoDecoder

def load_sample(index_path, sample_idx=0):
    """Load a single sample from WebDataset index."""
    with open(index_path) as f:
        index = json.load(f)

    if sample_idx >= len(index["entries"]):
        raise IndexError(f"Sample {sample_idx} out of range (max {len(index['entries'])-1})")

    entry = index["entries"][sample_idx]
    shard_path = index_path.parent / entry["shard"]

    print(f"Loading sample {sample_idx}:")
    print(f"  Shard: {entry['shard']}")
    print(f"  Key: {entry['key']}")

    # Decode video from tar
    import tarfile
    with tarfile.open(shard_path) as tar:
        # Get video
        video_member = tar.getmember(f"{entry['key']}/video.mp4")
        video_f = tar.extractfile(video_member)
        video_data = video_f.read()

        # Get metadata
        meta_member = tar.getmember(f"{entry['key']}/meta.json")
        meta_f = tar.extractfile(meta_member)
        meta = json.load(meta_f)

        # Get actions if present
        try:
            actions_member = tar.getmember(f"{entry['key']}/actions.npy")
            actions_f = tar.extractfile(actions_member)
            actions = np.load(actions_f)
        except KeyError:
            actions = None

    # Decode video frames
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp.write(video_data)
        tmp_path = tmp.name

    decoder = VideoDecoder(tmp_path)
    frames = []
    for frame in decoder:
        frames.append(frame.cpu().numpy().transpose(1, 2, 0))
    frames = np.array(frames)
    Path(tmp_path).unlink()

    return {
        "frames": frames,
        "actions": actions,
        "meta": meta,
        "entry": entry,
    }

def plot_3d_sample(sample, title="MIRA Sample"):
    """Create 3D visualization of sample."""
    frames = sample["frames"]
    actions = sample["actions"]
    meta = sample["meta"]
    n_frames = len(frames)

    fig = plt.figure(figsize=(16, 6))

    # Left: 3D frame grid
    ax1 = fig.add_subplot(121, projection="3d")
    frame_indices = np.linspace(0, n_frames - 1, min(16, n_frames), dtype=int)

    for i, fidx in enumerate(frame_indices):
        frame = frames[fidx]
        # Normalize frame
        frame_norm = frame.astype(float) / 255.0
        # Downsample for faster rendering
        frame_small = frame_norm[::4, ::4]
        # Create 3D voxel at this time position
        z_pos = fidx / (n_frames - 1) if n_frames > 1 else 0
        ax1.imshow(frame_small, extent=[i - 0.4, i + 0.4, 0, 1], zs=z_pos, zdir="z", alpha=0.6)

    ax1.set_xlabel("Frame (sampled)")
    ax1.set_ylabel("Pixel Y")
    ax1.set_zlabel("Time (normalized)")
    ax1.set_title(f"{title}\n{n_frames} frames @ {meta.get('fps', 20)} fps")

    # Right: Actions over time
    ax2 = fig.add_subplot(122)
    if actions is not None:
        action_names = ["W", "A", "S", "D", "Q", "E", "Space", "LShift", "LCtrl"]
        for i, name in enumerate(action_names[:actions.shape[1]]):
            ax2.plot(actions[:, i], label=name, alpha=0.7, linewidth=2)
        ax2.set_xlabel("Frame")
        ax2.set_ylabel("Action strength")
        ax2.set_title("Keyboard Input Timeline")
        ax2.legend(loc="upper right", fontsize=8)
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, "No actions recorded", ha="center", va="center", fontsize=12)
        ax2.set_xticks([])
        ax2.set_yticks([])

    plt.tight_layout()
    return fig

def main():
    parser = argparse.ArgumentParser(description="3D visualize MIRA WebDataset samples")
    parser.add_argument("--data", required=True, help="Path to index.json")
    parser.add_argument("--sample", type=int, default=0, help="Sample index (default 0)")
    parser.add_argument("--output", help="Save figure to file instead of showing")
    args = parser.parse_args()

    try:
        sample = load_sample(args.data, args.sample)
        print(f"  Shape: {sample['frames'].shape}")
        print(f"  Duration: {sample['frames'].shape[0] / sample['meta'].get('fps', 20):.1f}s")
        print()

        fig = plot_3d_sample(sample, title=f"Sample {args.sample}")

        if args.output:
            fig.savefig(args.output, dpi=100, bbox_inches="tight")
            print(f"✓ Saved to {args.output}")
        else:
            plt.show()

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

if __name__ == "__main__":
    main()
