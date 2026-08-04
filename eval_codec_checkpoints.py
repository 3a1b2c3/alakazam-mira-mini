#!/usr/bin/env python3
"""Evaluate and compare multiple codec checkpoints"""

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

try:
    from mira.codec.codec_model import VideoCodec
    from torchcodec.decoders import VideoDecoder
except ImportError:
    print("ERROR: mira not in PYTHONPATH")
    sys.exit(1)

def load_video_clip(shard_path: Path, match_id: str) -> torch.Tensor:
    """Load video from tar"""
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
            frames = torch.stack([f.cpu() for f in dec])
        finally:
            try:
                Path(path).unlink()
            except OSError:
                pass
        return frames
    except Exception as e:
        return None

def eval_checkpoint(codec_path: str, data_index: Path, num_clips: int = 32) -> dict:
    """Evaluate codec checkpoint"""
    try:
        codec = VideoCodec.load_from_checkpoint(codec_path, device="cuda").eval()
    except Exception as e:
        return {"error": str(e)}

    idx = json.load(open(data_index))
    root = data_index.parent
    entries = idx["entries"][:num_clips]

    psnrs = []
    for e in entries:
        shard = root / e["shard"]
        frames = load_video_clip(shard, e["match_id"])
        if frames is None:
            continue

        video = (frames.float() / 255.0).unsqueeze(0).to("cuda")
        tgt_h, tgt_w = codec.config.encoder.video.height, codec.config.encoder.video.width

        with torch.no_grad():
            import torch.nn.functional as F
            from einops import rearrange

            b, t = video.shape[:2]
            video_resized = F.interpolate(
                rearrange(video, "b t c h w -> (b t) c h w"),
                size=(tgt_h, tgt_w), mode="bilinear", align_corners=False
            )
            video_resized = rearrange(video_resized, "(b t) c h w -> b t c h w", b=b, t=t)

            input_norm, enc = codec.encode(video_resized)
            recon = codec.decode(enc.z)

            tt = min(input_norm.shape[1], recon.shape[1])
            mse = torch.nn.functional.mse_loss(recon[:, :tt], input_norm[:, :tt]).item()
            psnr = -10.0 * math.log10(mse + 1e-12)
            psnrs.append(psnr)

    if psnrs:
        return {
            "mean_psnr": sum(psnrs) / len(psnrs),
            "min_psnr": min(psnrs),
            "max_psnr": max(psnrs),
            "num_clips": len(psnrs)
        }
    return {"error": "no clips evaluated"}

def main():
    data_index = Path(r"C:\recordings\mira_wds\train\index.json")
    checkpoints = [
        "outputs/codec_ckpt_7000.pth",
        # Add more paths as needed
    ]

    if not data_index.exists():
        print(f"ERROR: {data_index} not found")
        return False

    print("=" * 60)
    print("Codec Checkpoint Evaluation")
    print("=" * 60)

    for ckpt_path in checkpoints:
        if not Path(ckpt_path).exists():
            print(f"\n{ckpt_path}: NOT FOUND")
            continue

        print(f"\nEvaluating {ckpt_path}...")
        result = eval_checkpoint(ckpt_path, data_index, num_clips=32)

        if "error" in result:
            print(f"  ERROR: {result['error']}")
        else:
            print(f"  Mean PSNR:  {result['mean_psnr']:.2f} dB")
            print(f"  Min PSNR:   {result['min_psnr']:.2f} dB")
            print(f"  Max PSNR:   {result['max_psnr']:.2f} dB")
            print(f"  Clips eval: {result['num_clips']}")

    return True

if __name__ == "__main__":
    sys.exit(0 if main() else 1)
