#!/usr/bin/env python3
"""Visualize a MIRA WebDataset sample: video frames + key presses + physics.

Reads the three per-frame members the RacerX->mira packer writes into each tar
(see make_mira_sample.py / PHYSICS_FORMAT.md):
    <key>.p0.mp4            video frames
    <key>.p0.jsonl          actions: one {"keys":[...]} per frame (mira vocab)
    <key>.p0.physics.jsonl  physics: one FrameState per frame (cars[0] pose/vel)

Usage: python visualize_sample.py --data <dataset-dir>/index.json [--sample 0] [--output out.png]
Example: python visualize_sample.py --data C:\recordings\mira_wds\train\index.json --sample 0 --output sample0.png
"""
import argparse
import json
import tarfile
import tempfile
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpecFromSubplotSpec
from torchcodec.decoders import VideoDecoder

# mira action vocab (make_mira_sample.py): the columns of the key-press timeline.
ACTION_VOCAB = ["W", "A", "S", "D", "Q", "E", "Space", "LShiftKey", "LControlKey"]
ACTION_LABELS = ["W", "A", "S", "D", "Q", "E", "Spc", "LShift", "LCtrl"]


def _read_member_text(tar, name):
    """Return a tar member's UTF-8 text, or None if it's absent."""
    try:
        f = tar.extractfile(name)
    except KeyError:
        return None
    if f is None:
        return None
    return f.read().decode("utf-8")


def _parse_actions(text):
    """Sparse {"keys":[...]} JSONL -> (F, len(ACTION_VOCAB)) binary pressed matrix."""
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        keys = set(json.loads(line).get("keys") or [])
        rows.append([1 if k in keys else 0 for k in ACTION_VOCAB])
    return np.array(rows, dtype=np.int8) if rows else None


def _parse_physics(text):
    """FrameState JSONL -> (pos[F,3], vel[F,3]) for the local car (else cars[0])."""
    pos, vel = [], []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        cars = json.loads(line).get("cars") or []
        car = next((c for c in cars if c.get("is_local")), cars[0] if cars else None)
        if car is None:
            pos.append([np.nan] * 3)
            vel.append([np.nan] * 3)
            continue
        loc, v = car["location"], car["velocity"]
        pos.append([loc["x"], loc["y"], loc["z"]])
        vel.append([v["x"], v["y"], v["z"]])
    if not pos:
        return None, None
    return np.array(pos, dtype=np.float64), np.array(vel, dtype=np.float64)


def load_sample(index_path, sample_idx=0):
    """Load frames + actions + physics for a single WebDataset sample."""
    index_path = Path(index_path)
    if not index_path.exists():
        raise FileNotFoundError(f"Index not found: {index_path}\nExpect: <dataset>/index.json")
    if index_path.name != "index.json":
        raise ValueError(f"Expected index.json, got {index_path.name}\nUsage: --data <dataset-dir>/index.json")

    with open(index_path, encoding="utf-8") as f:
        index = json.load(f)
    if sample_idx >= len(index["entries"]):
        raise IndexError(f"Sample {sample_idx} out of range (max {len(index['entries']) - 1})")

    entry = index["entries"][sample_idx]
    shard_path = index_path.parent / entry["shard"]

    print(f"Loading sample {sample_idx}:")
    print(f"  Match: {entry['match_id']}")
    print(f"  Shard: {entry['shard']}")
    print(f"  Players: {entry['n_players']}, Frames: {sum(entry['chunk_frames'])}")

    with tarfile.open(shard_path) as tar:
        match_prefix = entry["match_id"]
        video_file = next(
            (m.name for m in tar.getmembers()
             if match_prefix in m.name and m.name.endswith(".p0.mp4")),
            None,
        )
        if not video_file:
            raise FileNotFoundError(f"No .p0.mp4 found for match {match_prefix}")
        # Sibling members share the same "<key>.p0" base as the video.
        base = video_file[: -len(".mp4")]  # "<key>.p0"
        video_data = tar.extractfile(video_file).read()
        actions_text = _read_member_text(tar, f"{base}.jsonl")
        physics_text = _read_member_text(tar, f"{base}.physics.jsonl")

    actions = _parse_actions(actions_text) if actions_text else None
    pos, vel = _parse_physics(physics_text) if physics_text else (None, None)
    meta = {"fps": 20, "duration": sum(entry["chunk_frames"]) / 20}

    # Decode video frames (torchcodec needs a file path).
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp.write(video_data)
        tmp_path = tmp.name
    try:
        decoder = VideoDecoder(tmp_path)
        frames = np.array([fr.cpu().numpy().transpose(1, 2, 0) for fr in decoder])
        del decoder
    finally:
        try:
            Path(tmp_path).unlink()
        except (PermissionError, OSError):
            pass

    print(f"  actions: {'none' if actions is None else actions.shape[0]} frames"
          f" | physics: {'none' if pos is None else pos.shape[0]} frames")
    return {"frames": frames, "actions": actions, "pos": pos, "vel": vel, "meta": meta, "entry": entry}


def plot_sample(sample, title="MIRA Sample"):
    """Frames strip (top) + key-press timeline (middle) + physics (bottom)."""
    frames = sample["frames"]
    actions = sample["actions"]
    pos, vel = sample["pos"], sample["vel"]
    fps = sample["meta"].get("fps", 20)
    n_frames = len(frames)

    fig = plt.figure(figsize=(16, 11))
    gs = fig.add_gridspec(3, 4, height_ratios=[1.0, 0.8, 1.3], hspace=0.35, wspace=0.55)

    # -- Row 0: frame strip -------------------------------------------------
    n_show = min(8, n_frames)
    strip = GridSpecFromSubplotSpec(1, n_show, subplot_spec=gs[0, :], wspace=0.05)
    for i, fidx in enumerate(np.linspace(0, n_frames - 1, n_show, dtype=int)):
        ax = fig.add_subplot(strip[0, i])
        ax.imshow(frames[fidx].astype(np.uint8))
        ax.set_title(f"f{fidx}", fontsize=8)
        ax.axis("off")

    # -- Row 1: key-press timeline (heatmap: keys x frames) -----------------
    ax_keys = fig.add_subplot(gs[1, :])
    if actions is not None and actions.shape[0] > 0:
        ax_keys.imshow(actions.T, aspect="auto", cmap="Greens", vmin=0, vmax=1,
                       interpolation="nearest", extent=[0, actions.shape[0], len(ACTION_VOCAB), 0])
        ax_keys.set_yticks(np.arange(len(ACTION_LABELS)) + 0.5)
        ax_keys.set_yticklabels(ACTION_LABELS, fontsize=9)
        ax_keys.set_xlabel("Frame")
        ax_keys.set_title("Key presses (green = held)")
        held = actions.sum(axis=0)
        used = ", ".join(f"{ACTION_LABELS[i]}:{int(held[i])}" for i in range(len(held)) if held[i])
        ax_keys.text(1.005, 0.5, used or "no keys", transform=ax_keys.transAxes,
                     fontsize=7, va="center", rotation=90, color="dimgray")
    else:
        ax_keys.text(0.5, 0.5, "No actions in tar (.p0.jsonl)", ha="center", va="center", fontsize=12)
        ax_keys.set_xticks([]); ax_keys.set_yticks([])

    # -- Row 2 left: top-down trajectory ------------------------------------
    ax_traj = fig.add_subplot(gs[2, :2])
    if pos is not None and np.isfinite(pos).any():
        t = np.arange(len(pos))
        sc = ax_traj.scatter(pos[:, 0], pos[:, 1], c=t, cmap="viridis", s=8)
        ax_traj.plot(pos[:, 0], pos[:, 1], color="gray", lw=0.5, alpha=0.5)
        ax_traj.scatter(pos[0, 0], pos[0, 1], c="lime", s=90, marker="o", edgecolors="k", label="start", zorder=5)
        ax_traj.scatter(pos[-1, 0], pos[-1, 1], c="red", s=90, marker="X", edgecolors="k", label="end", zorder=5)
        ax_traj.set_aspect("equal", adjustable="datalim")
        ax_traj.set_xlabel("x (native units)"); ax_traj.set_ylabel("y (native units)")
        ax_traj.set_title("Top-down path (color = time)")
        ax_traj.legend(loc="best", fontsize=8); ax_traj.grid(True, alpha=0.3)
        fig.colorbar(sc, ax=ax_traj, label="frame", fraction=0.046, pad=0.04)
    else:
        ax_traj.text(0.5, 0.5, "No physics in tar (.p0.physics.jsonl)", ha="center", va="center", fontsize=12)
        ax_traj.set_xticks([]); ax_traj.set_yticks([])

    # -- Row 2 right: speed over time ---------------------------------------
    ax_spd = fig.add_subplot(gs[2, 2:])
    if pos is not None and np.isfinite(pos).any():
        spd_v = np.linalg.norm(vel, axis=1)
        spd_d = np.concatenate([[0.0], np.linalg.norm(np.diff(pos, axis=0), axis=1) * fps])
        t = np.arange(len(pos))
        if np.nanmax(spd_v) > 1e-6:
            ax_spd.plot(t, spd_v, color="tab:blue", lw=1.5, label="|velocity|")
        ax_spd.plot(t, spd_d, color="tab:orange", lw=1.0, ls="--", alpha=0.8, label="pos-delta speed")
        ax_spd.set_xlabel("Frame"); ax_spd.set_ylabel("speed (native units/s)")
        ax_spd.set_title("Speed"); ax_spd.legend(loc="best", fontsize=8); ax_spd.grid(True, alpha=0.3)
    else:
        ax_spd.text(0.5, 0.5, "No physics", ha="center", va="center", fontsize=12)
        ax_spd.set_xticks([]); ax_spd.set_yticks([])

    fig.suptitle(f"{title} | {n_frames} frames @ {fps} fps", fontsize=13, fontweight="bold")
    return fig


def main():
    parser = argparse.ArgumentParser(description="Visualize a MIRA WebDataset sample (frames + keys + physics)")
    parser.add_argument("--data", required=True, help="Path to index.json")
    parser.add_argument("--sample", type=int, default=0, help="Sample index (default 0)")
    parser.add_argument("--output", help="Save figure to file instead of showing")
    args = parser.parse_args()

    try:
        sample = load_sample(args.data, args.sample)
        print(f"  Shape: {sample['frames'].shape}")
        print(f"  Duration: {sample['frames'].shape[0] / sample['meta'].get('fps', 20):.1f}s")
        print()
        fig = plot_sample(sample, title=f"Sample {args.sample}")
        if args.output:
            fig.savefig(args.output, dpi=100, bbox_inches="tight")
            print(f"Saved to {args.output}")
        else:
            plt.show()
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        exit(1)


if __name__ == "__main__":
    main()
