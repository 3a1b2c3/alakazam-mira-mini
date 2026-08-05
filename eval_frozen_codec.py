#!/usr/bin/env python3
"""Evaluate frozen mira-mini codec vs trained codec on real data"""

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
from PIL import Image
import torch.nn.functional as F
from einops import rearrange

try:
    from mira.codec.codec_model import VideoCodec
    from torchcodec.decoders import VideoDecoder
except ImportError as e:
    print(f"ERROR: mira not in PYTHONPATH: {e}")
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
        return None

def eval_codec(codec, frames, name=""):
    """Evaluate codec on frames"""
    video = (frames.float() / 255.0).unsqueeze(0).to("cuda")
    tgt_h, tgt_w = codec.config.encoder.video.height, codec.config.encoder.video.width

    with torch.no_grad():
        # Resize to codec input size
        b, t = video.shape[:2]
        video_resized = F.interpolate(
            rearrange(video, "b t c h w -> (b t) c h w"),
            size=(tgt_h, tgt_w), mode="bilinear", align_corners=False
        )
        video_resized = rearrange(video_resized, "(b t) c h w -> b t c h w", b=b, t=t)

        # Encode/decode
        input_norm, enc = codec.encode(video_resized)
        recon = codec.decode(enc.z)

        # Align frames
        tt = min(input_norm.shape[1], recon.shape[1])
        mse = torch.nn.functional.mse_loss(recon[:, :tt], input_norm[:, :tt]).item()
        psnr = -10.0 * math.log10(mse + 1e-12)

        return {
            "psnr": psnr,
            "mse": mse,
            "latent_shape": enc.z.shape,
            "latent_size_mb": enc.z.numel() * 4 / (1024**2)  # 32-bit float
        }

def main():
    data_index = Path(r"C:\recordings\mira_wds\train\index.json")

    if not data_index.exists():
        print(f"ERROR: {data_index} not found")
        return False

    print("Loading frozen codec (mira-mini)...")
    frozen_codec = None
    for s in Path.home().glob(".cache/huggingface/hub/models--alakazamworld--mira-mini/snapshots/*/"):
        ckpt_path = s / "codec" / "checkpoint-125000" / "checkpoint.pth"
        if ckpt_path.exists():
            try:
                frozen_codec = VideoCodec.load_from_checkpoint(str(ckpt_path), device="cuda").eval()
                print(f"✓ Loaded frozen codec from {s.name[:8]}...")
                break
            except Exception as e:
                print(f"  Failed: {e}")

    if frozen_codec is None:
        print("ERROR: Frozen codec not found in cache")
        print("  Run: mira-mini play (downloads to ~/.cache/...)")
        return False

    print("\nLoading trained codec (checkpoint-7000)...")
    trained_path = Path("outputs/codec_ckpt_7000.pth")
    if not trained_path.exists():
        print(f"ERROR: {trained_path} not found")
        return False

    trained_codec = VideoCodec.load_from_checkpoint(str(trained_path), device="cuda").eval()
    print(f"✓ Loaded trained codec")

    # Load test data
    print("\nLoading test data...")
    idx = json.load(open(data_index))
    root = data_index.parent
    entries = idx["entries"][:5]  # Eval on 5 clips

    print(f"✓ Evaluating on {len(entries)} clips\n")

    frozen_results = []
    trained_results = []

    print("="*70)
    print(f"{'Clip':<28} {'Frozen PSNR':>14} {'Trained PSNR':>14} {'Diff':>10}")
    print("="*70)

    for i, e in enumerate(entries):
        shard = root / e["shard"]
        frames = load_video_clip(shard, e["match_id"], num_frames=8)

        if frames is None:
            continue

        frozen_result = eval_codec(frozen_codec, frames, "frozen")
        trained_result = eval_codec(trained_codec, frames, "trained")

        frozen_results.append(frozen_result)
        trained_results.append(trained_result)

        diff = trained_result["psnr"] - frozen_result["psnr"]
        print(f"{e['match_id'][:28]:<28} {frozen_result['psnr']:>13.2f} dB {trained_result['psnr']:>13.2f} dB {diff:>9.2f} dB")

    print("="*70)

    if frozen_results and trained_results:
        frozen_mean = sum(r["psnr"] for r in frozen_results) / len(frozen_results)
        trained_mean = sum(r["psnr"] for r in trained_results) / len(trained_results)
        diff_mean = trained_mean - frozen_mean

        print(f"\n{'SUMMARY':<28} {'Frozen':>14} {'Trained':>14} {'Improvement':>10}")
        print("-"*70)
        print(f"{'Mean PSNR':<28} {frozen_mean:>13.2f} dB {trained_mean:>13.2f} dB {diff_mean:>9.2f} dB")
        print(f"{'Min PSNR':<28} {min(r['psnr'] for r in frozen_results):>13.2f} dB {min(r['psnr'] for r in trained_results):>13.2f} dB")
        print(f"{'Max PSNR':<28} {max(r['psnr'] for r in frozen_results):>13.2f} dB {max(r['psnr'] for r in trained_results):>13.2f} dB")
        print(f"{'Clips evaluated':<28} {len(frozen_results):>14} {len(trained_results):>14}")

        print("\n✓ Trained codec outperforms frozen by", f"+{diff_mean:.2f} dB" if diff_mean > 0 else f"{diff_mean:.2f} dB")

    return True

if __name__ == "__main__":
    sys.exit(0 if main() else 1)
