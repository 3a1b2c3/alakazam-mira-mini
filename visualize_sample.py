#!/usr/bin/env python3
"""3D visualization of MIRA WebDataset samples.
Shows video frames + actions in interactive 3D space.
Usage: python visualize_sample.py --data <dataset-dir>/index.json [--sample 0] [--output out.png]
Example: python visualize_sample.py --data C:\recordings\mira_wds\train\index.json --sample 0 --output sample0.png
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
    index_path = Path(index_path)
    if not index_path.exists():
        raise FileNotFoundError(f"Index not found: {index_path}\nExpect: <dataset>/index.json")
    if index_path.name != "index.json":
        raise ValueError(f"Expected index.json, got {index_path.name}\nUsage: --data <dataset-dir>/index.json")

    with open(index_path, encoding="utf-8") as f:
        index = json.load(f)

    if sample_idx >= len(index["entries"]):
        raise IndexError(f"Sample {sample_idx} out of range (max {len(index['entries'])-1})")

    entry = index["entries"][sample_idx]
    shard_path = index_path.parent / entry["shard"]

    print(f"Loading sample {sample_idx}:")
    print(f"  Match: {entry['match_id']}")
    print(f"  Shard: {entry['shard']}")
    print(f"  Players: {entry['n_players']}, Frames: {sum(entry['chunk_frames'])}")

    # Decode video from tar - RacerX stores .p0.mp4 files directly
    import tarfile
    try:
        with tarfile.open(shard_path) as tar:
            # Find first .p0.mp4 file for this match
            match_prefix = entry['match_id']
            video_file = None

            for member in tar.getmembers():
                if match_prefix in member.name and member.name.endswith('.p0.mp4'):
                    video_file = member.name
                    break

            if not video_file:
                raise FileNotFoundError(f"No .p0.mp4 found for match {match_prefix}")

            # Extract video
            video_f = tar.extractfile(video_file)
            video_data = video_f.read()

            # Default metadata (RacerX doesn't include meta.json in tar)
            meta = {"fps": 20, "duration": sum(entry['chunk_frames']) / 20}

            # Actions are usually not in tar either (stored separately in dataset)
            actions = None

    except Exception as e:
        raise RuntimeError(f"Failed to load sample from tar: {e}")

    # Decode video frames
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp.write(video_data)
        tmp_path = tmp.name

    try:
        decoder = VideoDecoder(tmp_path)
        frames = []
        for frame in decoder:
            frames.append(frame.cpu().numpy().transpose(1, 2, 0))
        frames = np.array(frames)
        del decoder  # Explicitly close decoder
    finally:
        # Clean up temp file (ignore errors on Windows file locking)
        try:
            Path(tmp_path).unlink()
        except (PermissionError, OSError):
            pass

    return {
        "frames": frames,
        "actions": actions,
        "meta": meta,
        "entry": entry,
    }

def plot_3d_sample(sample, title="MIRA Sample"):
    """Visualize sample - frames grid + actions timeline."""
    frames = sample["frames"]
    actions = sample["actions"]
    meta = sample["meta"]
    n_frames = len(frames)

    fig = plt.figure(figsize=(16, 10))

    # Top: Sample frames grid
    n_show = min(16, n_frames)
    indices = np.linspace(0, n_frames - 1, n_show, dtype=int)

    for i, fidx in enumerate(indices):
        ax = fig.add_subplot(3, n_show // 3 + 1, i + 1)
        frame = frames[fidx].astype(np.uint8)
        ax.imshow(frame)
        ax.set_title(f"Frame {fidx}", fontsize=8)
        ax.axis("off")

    # Bottom: Actions timeline
    ax_actions = fig.add_subplot(3, 1, 3)
    if actions is not None and actions.shape[0] > 0:
        action_names = ["W", "A", "S", "D", "Q", "E", "Space", "LShift", "LCtrl"]
        for i in range(min(actions.shape[1], len(action_names))):
            ax_actions.plot(actions[:, i], label=action_names[i], alpha=0.7, linewidth=1.5)
        ax_actions.set_xlabel("Frame")
        ax_actions.set_ylabel("Action (pressed)")
        ax_actions.set_title("Keyboard Input Timeline")
        ax_actions.legend(loc="upper right", fontsize=8, ncol=3)
        ax_actions.grid(True, alpha=0.3)
    else:
        ax_actions.text(0.5, 0.5, "No actions recorded", ha="center", va="center", fontsize=12)
        ax_actions.set_xticks([])
        ax_actions.set_yticks([])

    fig.suptitle(f"{title} | {n_frames} frames @ {meta.get('fps', 20)} fps", fontsize=12, fontweight="bold")
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
