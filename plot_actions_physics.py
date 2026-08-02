#!/usr/bin/env python3
"""Plot keyboard actions vs physics response.
Shows how player input (keys) affects vehicle dynamics (position, velocity, rotation).
Usage: python plot_actions_physics.py --data <index.json> [--sample 0] [--output out.png]
"""
import json
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import tarfile

def load_sample_data(index_path, sample_idx=0):
    """Load actions and physics from sample."""
    index_path = Path(index_path)
    if not index_path.exists():
        raise FileNotFoundError(f"Index not found: {index_path}")

    with open(index_path, encoding="utf-8") as f:
        index = json.load(f)

    if sample_idx >= len(index.get("entries", [])):
        raise IndexError(f"Sample {sample_idx} out of range")

    entry = index["entries"][sample_idx]
    shard_path = index_path.parent / entry["shard"]

    print(f"Loading sample {sample_idx}: {entry['match_id']}")

    match_prefix = entry["match_id"]
    physics_data = []

    try:
        with tarfile.open(shard_path) as tar:
            # Find physics files
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

    return {"physics": physics_data, "entry": entry}

def plot_actions_physics(data):
    """Plot keyboard actions and physics response together."""
    physics = data["physics"]
    n_frames = len(physics)

    fig, axes = plt.subplots(4, 2, figsize=(16, 12))

    # Extract physics time series
    times = np.arange(n_frames)
    positions = {"x": [], "y": [], "z": []}
    velocities = {"x": [], "y": [], "z": []}
    speed = []
    rotation = []

    for frame_data in physics:
        # Position
        pos = frame_data.get("position", {})
        for axis in ["x", "y", "z"]:
            positions[axis].append(pos.get(axis, 0))

        # Velocity
        vel = frame_data.get("velocity", {})
        for axis in ["x", "y", "z"]:
            velocities[axis].append(vel.get(axis, 0))

        # Speed (magnitude of velocity)
        vx = vel.get("x", 0)
        vy = vel.get("y", 0)
        vz = vel.get("z", 0)
        speed.append(np.sqrt(vx**2 + vy**2 + vz**2))

        # Rotation (yaw/heading)
        rot = frame_data.get("rotation", {})
        rotation.append(rot.get("yaw", rot.get("z", 0)))

    # Convert to numpy
    for axis in ["x", "y", "z"]:
        positions[axis] = np.array(positions[axis])
        velocities[axis] = np.array(velocities[axis])
    speed = np.array(speed)
    rotation = np.array(rotation)

    # Plot position
    axes[0, 0].plot(times, positions["x"], label="X", color="r", alpha=0.7)
    axes[0, 0].plot(times, positions["y"], label="Y", color="g", alpha=0.7)
    axes[0, 0].plot(times, positions["z"], label="Z", color="b", alpha=0.7)
    axes[0, 0].set_ylabel("Position (m)")
    axes[0, 0].set_title("Vehicle Position (World)")
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].grid(True, alpha=0.3)

    # Plot velocity vector
    axes[0, 1].plot(times, velocities["x"], label="Vx", color="r", alpha=0.7)
    axes[0, 1].plot(times, velocities["y"], label="Vy", color="g", alpha=0.7)
    axes[0, 1].plot(times, velocities["z"], label="Vz", color="b", alpha=0.7)
    axes[0, 1].set_ylabel("Velocity (m/s)")
    axes[0, 1].set_title("Velocity Components")
    axes[0, 1].legend(fontsize=8)
    axes[0, 1].grid(True, alpha=0.3)

    # Plot speed magnitude
    axes[1, 0].plot(times, speed, linewidth=2, color="purple", alpha=0.7)
    axes[1, 0].fill_between(times, speed, alpha=0.3, color="purple")
    axes[1, 0].set_ylabel("Speed (m/s)")
    axes[1, 0].set_title("Vehicle Speed (Velocity Magnitude)")
    axes[1, 0].grid(True, alpha=0.3)

    # Plot rotation/heading
    axes[1, 1].plot(times, np.degrees(rotation), linewidth=2, color="orange", alpha=0.7)
    axes[1, 1].set_ylabel("Heading (degrees)")
    axes[1, 1].set_title("Vehicle Rotation (Yaw)")
    axes[1, 1].grid(True, alpha=0.3)

    # Plot inferred actions from physics (speed changes, direction changes)
    # Acceleration (derivative of speed)
    accel = np.diff(speed, prepend=speed[0])
    axes[2, 0].plot(times, accel, linewidth=1.5, color="darkred", alpha=0.7)
    axes[2, 0].axhline(y=0, color="k", linestyle="--", alpha=0.3)
    axes[2, 0].fill_between(times, accel, alpha=0.2, where=accel>0, color="green", label="Accelerating")
    axes[2, 0].fill_between(times, accel, alpha=0.2, where=accel<0, color="red", label="Decelerating")
    axes[2, 0].set_ylabel("Acceleration (m/s²)")
    axes[2, 0].set_title("Vehicle Acceleration (Speed derivative)")
    axes[2, 0].legend(fontsize=8)
    axes[2, 0].grid(True, alpha=0.3)

    # Plot steering (heading changes)
    heading_change = np.diff(rotation, prepend=rotation[0])
    axes[2, 1].plot(times, np.degrees(heading_change), linewidth=1.5, color="darkblue", alpha=0.7)
    axes[2, 1].axhline(y=0, color="k", linestyle="--", alpha=0.3)
    axes[2, 1].fill_between(times, np.degrees(heading_change), alpha=0.2, where=heading_change>0, color="orange", label="Right turn")
    axes[2, 1].fill_between(times, np.degrees(heading_change), alpha=0.2, where=heading_change<0, color="cyan", label="Left turn")
    axes[2, 1].set_ylabel("Heading Change (deg/frame)")
    axes[2, 1].set_title("Steering Rate (Yaw derivative)")
    axes[2, 1].legend(fontsize=8)
    axes[2, 1].grid(True, alpha=0.3)

    # Summary stats
    axes[3, 0].axis("off")
    stats = f"""Physics Summary (n={n_frames} frames)
Position range:
  X: [{positions['x'].min():.1f}, {positions['x'].max():.1f}] m
  Y: [{positions['y'].min():.1f}, {positions['y'].max():.1f}] m
  Z: [{positions['z'].min():.1f}, {positions['z'].max():.1f}] m

Velocity range:
  Speed: [{speed.min():.1f}, {speed.max():.1f}] m/s
  Accel: [{accel.min():.2f}, {accel.max():.2f}] m/s²

Rotation:
  Heading: [{np.degrees(rotation).min():.1f}°, {np.degrees(rotation).max():.1f}°]
  Turn rate: [{np.degrees(heading_change).min():.2f}, {np.degrees(heading_change).max():.2f}]°/frame"""
    axes[3, 0].text(0.05, 0.5, stats, fontsize=9, verticalalignment="center", family="monospace",
                   bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.5))

    # Legend for interpretation
    axes[3, 1].axis("off")
    legend_text = """How to read these plots:

Position: Where the vehicle is in 3D space
Velocity: Speed/direction components (XYZ)
Speed: Total movement magnitude (useful metric)
Heading: Rotation angle (0° = forward, ±180° = backward)

Acceleration: Positive = speeding up, Negative = slowing down
Steering: Positive = turning right, Negative = turning left

Action inference:
• High acceleration → W (forward) or brake released
• Negative acceleration → S (brake) or forward released
• Positive steering → A (left) or D (right) released
• Negative steering → D (right) or A (left) released"""
    axes[3, 1].text(0.05, 0.5, legend_text, fontsize=8, verticalalignment="center", family="monospace",
                   bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.5))

    fig.suptitle("Keyboard Actions → Physics Response (Inferred from dynamics)", fontsize=12, fontweight="bold")
    plt.tight_layout()
    return fig

def main():
    parser = argparse.ArgumentParser(description="Plot keyboard actions vs physics response")
    parser.add_argument("--data", required=True, help="Path to index.json")
    parser.add_argument("--sample", type=int, default=0, help="Sample index")
    parser.add_argument("--output", help="Save to file")
    args = parser.parse_args()

    try:
        data = load_sample_data(args.data, args.sample)
        fig = plot_actions_physics(data)
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
