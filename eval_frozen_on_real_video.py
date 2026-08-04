#!/usr/bin/env python3
"""Evaluate frozen codec decoder on real Rocket League video"""

import os
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["TORCH_COMPILE_DISABLE"] = "1"

import torch
import sys
import math
import re
import argparse
import tarfile
import tempfile
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

def load_video_file(video_path: str, num_frames: int = 16) -> torch.Tensor:
    """Load video from MP4 file"""
    video_path = Path(video_path)
    if not video_path.exists():
        print(f"ERROR: {video_path} not found")
        return None

    try:
        dec = VideoDecoder(str(video_path))
        frames = torch.stack([f.cpu() for f in dec])[:num_frames]
        return frames
    except Exception as e:
        print(f"ERROR: Failed to load video: {e}")
        return None


def load_video_any(path: str, num_frames: int = 16) -> torch.Tensor:
    """Load frames from a raw .mp4 OR the first .p0.mp4 clip inside a mira_wds .tar shard."""
    p = Path(path)
    if p.suffix.lower() == ".tar":
        with tarfile.open(p) as tar:
            member = next((m.name for m in tar.getmembers() if m.name.endswith(".p0.mp4")), None)
            if member is None:
                print(f"ERROR: no .p0.mp4 clip in {p}")
                return None
            data = tar.extractfile(member).read()
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp.write(data)
            tmp_path = tmp.name
        try:
            dec = VideoDecoder(tmp_path)
            frames = torch.stack([f.cpu() for f in dec])[:num_frames]
        finally:
            try:
                Path(tmp_path).unlink()
            except OSError:
                pass
        print(f"  (extracted {member} from {p.name})")
        return frames
    return load_video_file(path, num_frames)

def eval_codec_on_video(codec, frames, name="", save_frames=False, output_dir=None):
    """Evaluate codec on video frames"""
    video = (frames.float() / 255.0).unsqueeze(0).to("cuda")
    tgt_h, tgt_w = codec.config.encoder.video.height, codec.config.encoder.video.width

    print(f"\n{name}")
    print(f"  Input shape: {video.shape} (B, T, C, H, W)")
    print(f"  Target codec size: {tgt_h}x{tgt_w}")

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

        print(f"  Latent shape: {enc.z.shape}")
        print(f"  Reconstructed: {recon.shape}")
        print(f"  PSNR: {psnr:.2f} dB")
        print(f"  MSE: {mse:.6f}")

        # Save sample frames if requested
        if save_frames and output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            for i in range(min(3, tt)):
                # Original
                inp_frame = (input_norm[0, i] * 0.5 + 0.5).clamp(0, 1).cpu()
                inp_img = Image.fromarray((inp_frame.permute(1, 2, 0).numpy() * 255).astype('uint8'))
                inp_img.save(output_dir / f"{name.lower()}_{i:02d}_input.png")

                # Reconstructed
                rec_frame = (recon[0, i] * 0.5 + 0.5).clamp(0, 1).cpu()
                rec_img = Image.fromarray((rec_frame.permute(1, 2, 0).numpy() * 255).astype('uint8'))
                rec_img.save(output_dir / f"{name.lower()}_{i:02d}_recon.png")

            print(f"  ✓ Saved 3 frame comparisons to {output_dir}/")

        return {"psnr": psnr, "mse": mse}

def main():
    ap = argparse.ArgumentParser(description="Run a codec on real video (.mp4 or mira_wds .tar shard).")
    ap.add_argument("checkpoint", nargs="?", default="outputs/codec_ckpt_7000.pth",
                    help="trained codec checkpoint.pth")
    ap.add_argument("--video", default=r"C:\workspace\world\mira\dataset_00000\2026-05-08T20-37-10Z-5f298d_c00000.p0.mp4",
                    help="video .mp4 OR mira_wds .tar shard (uses its first .p0.mp4 clip)")
    ap.add_argument("--no-frozen", action="store_true",
                    help="skip the frozen 125k codec; only run the trained one")
    args = ap.parse_args()

    video_path = args.video
    if not Path(video_path).exists():
        print(f"ERROR: {video_path} not found")
        return False

    print("="*70)
    print("Codec Reconstruction on Real Video")
    print("="*70)

    print(f"\nLoading video: {Path(video_path).name}")
    frames = load_video_any(video_path, num_frames=16)

    if frames is None:
        return False

    print(f"✓ Loaded {frames.shape[0]} frames at {frames.shape[2:]} resolution")

    # Load frozen codec (unless --no-frozen)
    frozen_codec = None
    if not args.no_frozen:
        print("\nLoading frozen codec (mira-mini checkpoint-125000)...")
        for s in Path.home().glob(".cache/huggingface/hub/models--alakazamworld--mira-mini/snapshots/*/"):
            ckpt_path = s / "codec" / "checkpoint-125000" / "checkpoint.pth"
            if ckpt_path.exists():
                try:
                    frozen_codec = VideoCodec.load_from_checkpoint(str(ckpt_path), device="cuda").eval()
                    print("✓ Loaded frozen codec")
                    break
                except Exception as e:
                    print(f"  Failed: {e}")
        if frozen_codec is None:
            print("  (frozen codec not found; continuing with trained only)")

    # Load trained codec.
    trained_codec = None
    trained_path = Path(args.checkpoint)
    _m = re.search(r"(\d+)", trained_path.stem)
    step = _m.group(1) if _m else "trained"
    if trained_path.exists():
        print(f"\nLoading trained codec ({trained_path.name})...")
        try:
            trained_codec = VideoCodec.load_from_checkpoint(str(trained_path), device="cuda").eval()
            print(f"✓ Loaded trained codec")
        except Exception as e:
            print(f"  Failed: {e}")

    # Evaluate
    print("\n" + "="*70)
    print("Evaluation Results")
    print("="*70)

    output_dir = Path(f"outputs/trained_{step}") if frozen_codec is None else Path(f"outputs/frozen_125000_vs_trained_{step}")
    frozen_result = None
    if frozen_codec is not None:
        frozen_result = eval_codec_on_video(
            frozen_codec, frames, "Frozen Codec",
            save_frames=True, output_dir=output_dir
        )

    if trained_codec:
        trained_result = eval_codec_on_video(
            trained_codec, frames, f"Trained Codec ({step})",
            save_frames=True, output_dir=output_dir
        )

        if frozen_result is not None:
            print("\n" + "="*70)
            print("Comparison")
            print("="*70)
            diff = trained_result["psnr"] - frozen_result["psnr"]
            print(f"Frozen:  {frozen_result['psnr']:.2f} dB")
            print(f"Trained: {trained_result['psnr']:.2f} dB")
            print(f"Improvement: {diff:+.2f} dB")

    print(f"\n✓ Output saved to {output_dir}/")
    return True

if __name__ == "__main__":
    sys.exit(0 if main() else 1)
