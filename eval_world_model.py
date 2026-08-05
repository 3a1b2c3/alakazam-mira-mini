#!/usr/bin/env python3
"""Evaluate world model checkpoint"""

import os
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["TORCH_COMPILE_DISABLE"] = "1"

import torch
import sys
import json
import math
import tempfile
import tarfile
from pathlib import Path

# Add mira to path
sys.path.insert(0, str(Path(__file__).parent.parent / "mira" / "src"))

try:
    from torchcodec.decoders import VideoDecoder
    # Try loading checkpoint without full module
    print("Note: World model evaluation requires mira repo in ../mira/")
except ImportError as e:
    print(f"ERROR: torchcodec not available: {e}")
    sys.exit(1)

def load_video_clip(shard_path: Path, match_id: str, num_frames: int = 16) -> torch.Tensor:
    """Load first N frames from video"""
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
        return None

def eval_world_model(ckpt_path: str, data_index: Path, num_clips: int = 5) -> dict:
    """Evaluate world model checkpoint"""

    if not Path(ckpt_path).exists():
        return {"error": f"Checkpoint not found: {ckpt_path}"}

    print(f"Loading world model from {ckpt_path}...")
    try:
        wm = WorldModel.load_from_checkpoint(ckpt_path, device="cuda").eval()
        print("✓ World model loaded")
    except Exception as e:
        return {"error": f"Failed to load: {e}"}

    idx = json.load(open(data_index))
    root = data_index.parent
    entries = idx["entries"][:num_clips]

    results = {"clips": []}

    for i, e in enumerate(entries):
        shard = root / e["shard"]
        frames = load_video_clip(shard, e["match_id"], num_frames=8)

        if frames is None:
            continue

        print(f"\n[{i+1}] {e['match_id'][:24]}")
        print(f"     Input: {frames.shape}")

        video = (frames.float() / 255.0).unsqueeze(0).to("cuda")  # (1, T, C, H, W)

        try:
            with torch.no_grad():
                # Encode first 4 frames, predict next 4
                context = video[:, :4]
                target = video[:, 4:8]

                # Run model (rollout)
                predictions = wm.predict(context, num_steps=4)
                print(f"     Predictions: {predictions.shape}")

                # Compute MSE/PSNR
                mse = torch.nn.functional.mse_loss(predictions, target).item()
                psnr = -10.0 * math.log10(mse + 1e-12)

                results["clips"].append({
                    "id": e["match_id"][:24],
                    "psnr": round(psnr, 2),
                    "mse": round(mse, 4)
                })
                print(f"     PSNR: {psnr:.2f} dB, MSE: {mse:.4f}")

        except Exception as e:
            print(f"     Error: {e}")
            import traceback
            traceback.print_exc()

    if results["clips"]:
        psnrs = [c["psnr"] for c in results["clips"]]
        results["mean_psnr"] = round(sum(psnrs) / len(psnrs), 2)
        results["min_psnr"] = round(min(psnrs), 2)
        results["max_psnr"] = round(max(psnrs), 2)
        results["num_clips"] = len(psnrs)

    return results

def main():
    ckpt_path = r"C:\workspace\world\alakazam-mira-mini\outputs\scratch_ch\checkpoint-21000-horde.pth"
    data_index = Path(r"C:\recordings\mira_wds\train\index.json")

    if not data_index.exists():
        print(f"ERROR: {data_index} not found")
        return False

    print("=" * 60)
    print("World Model Evaluation")
    print("=" * 60)

    result = eval_world_model(ckpt_path, data_index, num_clips=5)

    if "error" in result:
        print(f"\n✗ ERROR: {result['error']}")
        return False

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    if "mean_psnr" in result:
        print(f"Mean PSNR:  {result['mean_psnr']} dB")
        print(f"Min PSNR:   {result['min_psnr']} dB")
        print(f"Max PSNR:   {result['max_psnr']} dB")
        print(f"Clips eval: {result['num_clips']}")
    else:
        print("No clips evaluated")

    return True

if __name__ == "__main__":
    sys.exit(0 if main() else 1)
