#!/usr/bin/env python3
"""Combined visualization: video frames + actions + physics data.
Usage: python visualize_sample_full.py --data <index.json> [--sample 0] [--output out.png]
"""
import json
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import tarfile
from torchcodec.decoders import VideoDecoder
import tempfile

def load_sample(index_path, sample_idx=0):
    """Load video and physics from sample."""
    index_path = Path(index_path)
    if not index_path.exists():
        raise FileNotFoundError(f"Index not found: {index_path}")
    if index_path.name != "index.json":
        raise ValueError(f"Expected index.json, got {index_path.name}")

    with open(index_path, encoding="utf-8") as f:
        index = json.load(f)

    if sample_idx >= len(index.get("entries", [])):
        raise IndexError(f"Sample {sample_idx} out of range")

    entry = index["entries"][sample_idx]
    shard_path = index_path.parent / entry["shard"]

    print(f"Loading sample {sample_idx}: {entry['match_id']}")

    # Load video and physics from tar
    import tarfile
    match_prefix = entry["match_id"]
    video_data = None
    physics_data = []
    actions = None
    meta = {"fps": 20}

    try:
        with tarfile.open(shard_path) as tar:
            # Find video
            for member in tar.getmembers():
                if match_prefix in member.name and member.name.endswith(".p0.mp4"):
                    video_f = tar.extractfile(member)
                    video_data = video_f.read()
                    break

            # Find physics (.p0.physics.jsonl format)
            physics_files = [m for m in tar.getmembers()
                           if match_prefix in m.name and m.name.endswith(".p0.physics.jsonl")]

            if not physics_files:
                raise FileNotFoundError(f"No physics files found for {match_prefix} (looking for .p0.physics.jsonl)")

            for member in physics_files:
                physics_f = tar.extractfile(member)
                for line in physics_f:
                    try:
                        physics_data.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass

    except Exception as e:
        raise RuntimeError(f"Failed to load from tar: {e}")

    # Decode video
    frames = []
    if video_data:
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp.write(video_data)
            tmp_path = tmp.name

        try:
            decoder = VideoDecoder(tmp_path)
            for frame in decoder:
                frames.append(frame.cpu().numpy().transpose(1, 2, 0))
            frames = np.array(frames)
            del decoder
        finally:
            try:
                Path(tmp_path).unlink()
            except (PermissionError, OSError):
                pass

    return {
        "frames": frames,
        "physics": physics_data,
        "meta": meta,
        "entry": entry,
    }

def plot_combined(sample):
    """Plot video + physics together."""
    frames = sample["frames"]
    physics = sample["physics"]
    entry = sample["entry"]
    n_frames = len(frames)

    fig = plt.figure(figsize=(16, 12))

    # Row 1: Sample frames
    n_show = min(12, n_frames)
    indices = np.linspace(0, n_frames - 1, n_show, dtype=int)

    for i, fidx in enumerate(indices):
        ax = fig.add_subplot(4, n_show, i + 1)
        frame = frames[fidx].astype(np.uint8)
        ax.imshow(frame)
        ax.set_title(f"Frame {fidx}", fontsize=7)
        ax.axis("off")

    # Row 2: Physics position
    ax_pos = fig.add_subplot(4, 2, 2 * n_show // 6 + 1)
    if physics:
        for axis, color in zip(["x", "y", "z"], ["r", "g", "b"]):
            pos_vals = [p.get("position", {}).get(axis, 0) for p in physics]
            if any(pos_vals):
                ax_pos.plot(pos_vals, label=f"Pos {axis.upper()}", color=color, alpha=0.7)
        ax_pos.set_ylabel("Position (m)")
        ax_pos.set_title("Vehicle Position")
        ax_pos.legend(fontsize=8)
        ax_pos.grid(True, alpha=0.3)

    # Row 2: Physics velocity
    ax_vel = fig.add_subplot(4, 2, 2 * n_show // 6 + 2)
    if physics:
        for axis, color in zip(["x", "y", "z"], ["r", "g", "b"]):
            vel_vals = [p.get("velocity", {}).get(axis, 0) for p in physics]
            if any(vel_vals):
                ax_vel.plot(vel_vals, label=f"Vel {axis.upper()}", color=color, alpha=0.7)
        ax_vel.set_ylabel("Velocity (m/s)")
        ax_vel.set_title("Vehicle Velocity")
        ax_vel.legend(fontsize=8)
        ax_vel.grid(True, alpha=0.3)

    # Row 3: Physics rotation
    ax_rot = fig.add_subplot(4, 2, 2 * n_show // 6 + 3)
    if physics:
        rot_vals = [p.get("rotation", {}).get("yaw", p.get("rotation", {}).get("z", 0)) for p in physics]
        if any(rot_vals):
            ax_rot.plot(rot_vals, linewidth=2, color="purple", alpha=0.7)
        ax_rot.set_ylabel("Yaw (rad)")
        ax_rot.set_title("Vehicle Rotation (Heading)")
        ax_rot.grid(True, alpha=0.3)

    # Row 4: Summary
    ax_text = fig.add_subplot(4, 1, 4)
    ax_text.axis("off")
    info = f"""Match: {entry['match_id']}
Frames: {n_frames} @ {sample['meta'].get('fps', 20)} fps | Duration: {n_frames / sample['meta'].get('fps', 20):.1f}s
Players: {entry['n_players']} | Physics frames: {len(physics)}
Shard: {entry['shard']}"""
    ax_text.text(0.05, 0.5, info, fontsize=10, verticalalignment="center", family="monospace",
                 bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    fig.suptitle(f"Sample {sample} - Video + Physics", fontsize=12, fontweight="bold")
    plt.tight_layout()
    return fig

def main():
    parser = argparse.ArgumentParser(description="Visualize sample with video + physics")
    parser.add_argument("--data", required=True, help="Path to index.json")
    parser.add_argument("--sample", type=int, default=0, help="Sample index")
    parser.add_argument("--output", help="Save to file")
    args = parser.parse_args()

    try:
        sample = load_sample(args.data, args.sample)
        fig = plot_combined(args.sample)
        if args.output:
            fig.savefig(args.output, dpi=100, bbox_inches="tight")
            print(f"✓ Saved to {args.output}")
        else:
            plt.show()
    except Exception as e:
        print(f"✗ ERROR: {e}")
        exit(1)

if __name__ == "__main__":
    main()
