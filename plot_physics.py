#!/usr/bin/env python3
"""Plot Rocket League physics data (car position, velocity, boost).
Usage: python plot_physics.py --data <index.json> [--sample 0] [--output out.png]
"""
import json
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import tarfile

def load_physics(index_path, sample_idx=0):
    """Load physics.jsonl from sample."""
    index_path = Path(index_path)
    if not index_path.exists():
        raise FileNotFoundError(f"Index not found: {index_path}")

    with open(index_path, encoding="utf-8") as f:
        idx = json.load(f)

    if sample_idx >= len(idx.get("entries", [])):
        raise IndexError(f"Sample {sample_idx} out of range")

    entry = idx["entries"][sample_idx]
    shard_path = index_path.parent / entry["shard"]

    print(f"Loading physics for sample {sample_idx}:")
    print(f"  Match: {entry['match_id']}")
    print(f"  Shard: {entry['shard']}")

    match_prefix = entry["match_id"]
    physics_data = []

    try:
        with tarfile.open(shard_path) as tar:
            physics_files = [m for m in tar.getmembers()
                           if match_prefix in m.name and m.name.endswith(".p0.physics.jsonl")]

            if not physics_files:
                raise FileNotFoundError(f"No physics files found for {match_prefix}")

            for member in physics_files:
                physics_f = tar.extractfile(member)
                for line in physics_f:
                    try:
                        physics_data.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass

    except Exception as e:
        raise RuntimeError(f"Failed to load: {e}")

    if not physics_data:
        raise ValueError("No physics data loaded")

    print(f"  Loaded {len(physics_data)} frames")
    return physics_data

def plot_physics(physics_data, title="Rocket League Physics"):
    """Plot car position, velocity, boost."""
    n_frames = len(physics_data)

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    times = np.arange(n_frames)
    pos_x, pos_y, pos_z = [], [], []
    vel_x, vel_y, vel_z = [], [], []
    boost = []

    # Extract car[0] data (local player)
    for frame in physics_data:
        if 'cars' not in frame or not frame['cars']:
            continue

        car = frame['cars'][0]
        loc = car.get('location', {})
        vel = car.get('velocity', {})

        pos_x.append(loc.get('x', 0))
        pos_y.append(loc.get('y', 0))
        pos_z.append(loc.get('z', 0))

        vel_x.append(vel.get('x', 0))
        vel_y.append(vel.get('y', 0))
        vel_z.append(vel.get('z', 0))

        boost.append(car.get('boost_amount', 0))

    pos_x, pos_y, pos_z = np.array(pos_x), np.array(pos_y), np.array(pos_z)
    vel_x, vel_y, vel_z = np.array(vel_x), np.array(vel_y), np.array(vel_z)
    boost = np.array(boost)
    speed = np.sqrt(vel_x**2 + vel_y**2 + vel_z**2)

    # Position
    axes[0, 0].plot(times, pos_x, label="X", color="r", linewidth=1.5, alpha=0.8)
    axes[0, 0].plot(times, pos_y, label="Y", color="g", linewidth=1.5, alpha=0.8)
    axes[0, 0].plot(times, pos_z, label="Z", color="b", linewidth=1.5, alpha=0.8)
    axes[0, 0].set_ylabel("Position (UU)")
    axes[0, 0].set_title("Car Position")
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].grid(True, alpha=0.3)

    # Velocity components
    axes[0, 1].plot(times, vel_x, label="Vx", color="r", linewidth=1.5, alpha=0.8)
    axes[0, 1].plot(times, vel_y, label="Vy", color="g", linewidth=1.5, alpha=0.8)
    axes[0, 1].plot(times, vel_z, label="Vz", color="b", linewidth=1.5, alpha=0.8)
    axes[0, 1].set_ylabel("Velocity (UU/s)")
    axes[0, 1].set_title("Velocity Components")
    axes[0, 1].legend(fontsize=8)
    axes[0, 1].grid(True, alpha=0.3)

    # Speed (magnitude)
    axes[1, 0].plot(times, speed, color="purple", linewidth=2, alpha=0.8)
    axes[1, 0].fill_between(times, speed, alpha=0.2, color="purple")
    axes[1, 0].set_ylabel("Speed (UU/s)")
    axes[1, 0].set_title("Car Speed")
    axes[1, 0].set_xlabel("Frame")
    axes[1, 0].grid(True, alpha=0.3)

    # Boost
    axes[1, 1].bar(times, boost, color="orange", alpha=0.7, width=1)
    axes[1, 1].set_ylabel("Boost Amount")
    axes[1, 1].set_title("Boost Level (0-100)")
    axes[1, 1].set_xlabel("Frame")
    axes[1, 1].set_ylim([0, 100])
    axes[1, 1].grid(True, alpha=0.3, axis='y')

    fig.suptitle(f"{title} | {n_frames} frames", fontsize=12, fontweight="bold")
    plt.tight_layout()
    return fig

def main():
    parser = argparse.ArgumentParser(description="Plot Rocket League physics")
    parser.add_argument("--data", required=True, help="Path to index.json")
    parser.add_argument("--sample", type=int, default=0, help="Sample index")
    parser.add_argument("--output", help="Save to file")
    args = parser.parse_args()

    try:
        physics = load_physics(args.data, args.sample)
        fig = plot_physics(physics)
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
