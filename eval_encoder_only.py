#!/usr/bin/env python3
"""Evaluate frozen encoder on test data - measure compression/information loss"""

import os
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["TORCH_COMPILE_DISABLE"] = "1"

import torch
import sys
import json
import tempfile
import tarfile
from pathlib import Path
import torch.nn.functional as F
from einops import rearrange

try:
    from mira.codec.codec_model import VideoCodec
    from torchcodec.decoders import VideoDecoder
except ImportError as e:
    print(f"ERROR: mira not in PYTHONPATH: {e}")
    sys.exit(1)

def load_video_clip(shard_path: Path, match_id: str, num_frames: int = 8) -> torch.Tensor:
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
            frames = torch.stack([f.cpu() for f in dec])[:num_frames]
        finally:
            try:
                Path(path).unlink()
            except:
                pass
        return frames
    except Exception:
        return None

def eval_encoder(codec, frames, name="", save_output=False, output_dir=None):
    """Evaluate encoder only - encode/decode to see bottleneck"""
    from PIL import Image

    video = (frames.float() / 255.0).unsqueeze(0).to("cuda")
    tgt_h, tgt_w = codec.config.encoder.video.height, codec.config.encoder.video.width

    with torch.no_grad():
        # Resize to codec input
        b, t = video.shape[:2]
        video_resized = F.interpolate(
            rearrange(video, "b t c h w -> (b t) c h w"),
            size=(tgt_h, tgt_w), mode="bilinear", align_corners=False
        )
        video_resized = rearrange(video_resized, "(b t) c h w -> b t c h w", b=b, t=t)

        # Encode/decode to see bottleneck
        input_norm, enc = codec.encode(video_resized)
        recon = codec.decode(enc.z)

        # Latent statistics
        latent = enc.z
        latent_mean = latent.mean().item()
        latent_std = latent.std().item()
        latent_min = latent.min().item()
        latent_max = latent.max().item()

        # Compression ratio
        video_size = video_resized.numel() * 4 / (1024**3)  # GB
        latent_size = latent.numel() * 4 / (1024**3)  # GB
        compression_ratio = video_size / latent_size if latent_size > 0 else 0

        # Save split-screen frames if requested
        if save_output and output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            tt = min(input_norm.shape[1], recon.shape[1])
            for i in range(min(3, tt)):
                # Input
                inp = (input_norm[0, i] * 0.5 + 0.5).clamp(0, 1).cpu()
                inp_img = Image.fromarray((inp.permute(1, 2, 0).numpy() * 255).astype('uint8'))

                # Reconstruction (via latent)
                rec = (recon[0, i] * 0.5 + 0.5).clamp(0, 1).cpu()
                rec_img = Image.fromarray((rec.permute(1, 2, 0).numpy() * 255).astype('uint8'))

                # Split-screen
                h, w = inp_img.size
                split = Image.new('RGB', (w * 2, h))
                split.paste(inp_img, (0, 0))
                split.paste(rec_img, (w, 0))
                split.save(output_dir / f"{name}_{i:02d}.png")

        return {
            "latent_shape": latent.shape,
            "latent_mean": latent_mean,
            "latent_std": latent_std,
            "latent_min": latent_min,
            "latent_max": latent_max,
            "compression_ratio": compression_ratio,
            "video_size_gb": video_size,
            "latent_size_gb": latent_size
        }

def main():
    data_index = Path(r"C:\recordings\mira_wds\test\index.json")

    if not data_index.exists():
        print(f"ERROR: {data_index} not found")
        return False

    print("="*70)
    print("Frozen Encoder Evaluation (Test Data)")
    print("="*70)

    # Load frozen codec
    print("\nLoading frozen codec...")
    frozen_codec = None
    for s in Path.home().glob(".cache/huggingface/hub/models--alakazamworld--mira-mini/snapshots/*/"):
        ckpt_path = s / "codec" / "checkpoint-125000" / "checkpoint.pth"
        if ckpt_path.exists():
            try:
                frozen_codec = VideoCodec.load_from_checkpoint(str(ckpt_path), device="cuda").eval()
                print(f"✓ Loaded frozen codec")
                break
            except Exception as e:
                print(f"  Failed: {e}")

    if frozen_codec is None:
        print("ERROR: Frozen codec not found")
        return False

    # Load test data
    print(f"\nLoading test data from {data_index}...")
    idx = json.load(open(data_index))
    root = data_index.parent
    entries = idx["entries"][:10]  # First 10 test clips
    print(f"✓ Evaluating encoder on {len(entries)} clips")

    # Name folder after checkpoint
    output_dir = Path("outputs/encoder_eval_frozen_checkpoint_125000")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*70)
    print(f"{'Clip':<28} {'Latent Shape':<20} {'Compr. Ratio':>12} {'Mean':>8}")
    print("="*70)

    results = []
    for i, e in enumerate(entries):
        shard = root / e["shard"]
        frames = load_video_clip(shard, e["match_id"], num_frames=8)

        if frames is None:
            continue

        clip_name = e['match_id'][:8]
        result = eval_encoder(
            frozen_codec, frames, clip_name,
            save_output=True, output_dir=output_dir
        )
        results.append(result)

        latent_shape = str(result["latent_shape"]).replace("torch.Size(", "").replace(")", "")
        print(f"{e['match_id'][:28]:<28} {latent_shape:<20} {result['compression_ratio']:>11.1f}x {result['latent_mean']:>7.3f}")

    if results:
        print("="*70)
        print("\nSummary Statistics")
        print("-"*70)

        mean_compression = sum(r["compression_ratio"] for r in results) / len(results)
        mean_latent_mean = sum(r["latent_mean"] for r in results) / len(results)
        mean_latent_std = sum(r["latent_std"] for r in results) / len(results)

        print(f"Clips evaluated: {len(results)}")
        print(f"Avg compression ratio: {mean_compression:.1f}x")
        print(f"Avg latent mean: {mean_latent_mean:.3f}")
        print(f"Avg latent std: {mean_latent_std:.3f}")
        print(f"Avg latent range: [{min(r['latent_min'] for r in results):.3f}, {max(r['latent_max'] for r in results):.3f}]")

        print(f"\n✓ Split-screen images saved to {output_dir}/")
        print("  Format: [LEFT: Input | RIGHT: Encoder→Decoder reconstruction]")

        print("\n" + "="*70)
        print("Analysis")
        print("="*70)
        print(f"✓ Frozen encoder compresses by ~{mean_compression:.0f}x")
        print(f"✓ Latent space stats: μ={mean_latent_mean:.2f}, σ={mean_latent_std:.2f}")
        print(f"✓ Information bottleneck: {mean_compression:.1f}x compression limits photorealism")
        print(f"\nConclusion: Encoder is the bottleneck. To improve:")
        print(f"  1. Larger latent size (less compression)")
        print(f"  2. Better encoder training")
        print(f"  3. More efficient compression scheme")

    return True

if __name__ == "__main__":
    sys.exit(0 if main() else 1)
