#!/usr/bin/env python3
"""Evaluate codecs on all videos in a WebDataset shard"""

import os
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["TORCH_COMPILE_DISABLE"] = "1"

import torch
import sys
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

def load_video_from_tar(tar_path: Path, num_frames: int = 8):
    """Extract first .mp4 from tar and load frames"""
    try:
        with tarfile.open(tar_path) as tar:
            mp4_members = [m for m in tar.getmembers() if m.name.endswith(".p0.mp4")]
            if not mp4_members:
                return None, None

            member = mp4_members[0]
            data = tar.extractfile(member).read()

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp.write(data)
            path = tmp.name

        try:
            dec = VideoDecoder(path)
            frames = torch.stack([f.cpu() for f in dec])[:num_frames]
            return frames, member.name.split("/")[0]  # Return clip ID
        finally:
            try:
                Path(path).unlink()
            except:
                pass
    except Exception as e:
        return None, None

def eval_codec(codec, frames, name="", save_images=False, output_dir=None, clip_id=""):
    """Evaluate codec"""
    if frames is None:
        return None

    video = (frames.float() / 255.0).unsqueeze(0).to("cuda")
    tgt_h, tgt_w = codec.config.encoder.video.height, codec.config.encoder.video.width

    with torch.no_grad():
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

        # Save split-screen frames
        if save_images and output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            for i in range(min(3, tt)):
                # Input
                inp = (input_norm[0, i] * 0.5 + 0.5).clamp(0, 1).cpu()
                inp_img = Image.fromarray((inp.permute(1, 2, 0).numpy() * 255).astype('uint8'))

                # Reconstruction
                rec = (recon[0, i] * 0.5 + 0.5).clamp(0, 1).cpu()
                rec_img = Image.fromarray((rec.permute(1, 2, 0).numpy() * 255).astype('uint8'))

                # Split-screen
                w, h = inp_img.size
                split = Image.new('RGB', (w * 2, h))
                split.paste(inp_img, (0, 0))
                split.paste(rec_img, (w, 0))
                split.save(output_dir / f"{clip_id}_{name}_frame_{i:02d}.png")

        return psnr

def main():
    shard_dir = Path(r"C:\recordings\mira_wds\test\000")

    if not shard_dir.exists():
        print(f"ERROR: {shard_dir} not found")
        return False

    # Find tar files
    tar_files = sorted(shard_dir.glob("*.tar"))
    if not tar_files:
        print(f"ERROR: No .tar files in {shard_dir}")
        return False

    print("="*70)
    print(f"Codec Evaluation on Shard: {shard_dir.name}")
    print("="*70)
    print(f"Found {len(tar_files)} tar files\n")

    # Load codecs
    print("Loading frozen codec...")
    frozen_codec = None
    for s in Path.home().glob(".cache/huggingface/hub/models--alakazamworld--mira-mini/snapshots/*/"):
        ckpt_path = s / "codec" / "checkpoint-125000" / "checkpoint.pth"
        if ckpt_path.exists():
            try:
                frozen_codec = VideoCodec.load_from_checkpoint(str(ckpt_path), device="cuda").eval()
                print("✓ Loaded frozen codec")
                break
            except Exception as e:
                print(f"  Failed: {e}")

    print("Loading trained codec (checkpoint-15000)...")
    trained_path = Path("outputs/codec_ckpt_15000.pth")
    trained_codec = None
    if trained_path.exists():
        try:
            trained_codec = VideoCodec.load_from_checkpoint(str(trained_path), device="cuda").eval()
            print("✓ Loaded trained codec (checkpoint-15000)")
        except Exception as e:
            print(f"  Failed: {e}")

    if not frozen_codec:
        print("ERROR: No codecs loaded")
        return False

    # Output directory for images
    output_dir = Path("outputs/shard_000_frozen_vs_trained_15000")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Evaluate
    print("\n" + "="*70)
    print(f"{'Clip':<24} {'Frozen (dB)':>14} {'Trained (dB)':>14} {'Diff':>10}")
    print("="*70)

    frozen_psnrs = []
    trained_psnrs = []

    for i, tar_file in enumerate(tar_files[:10]):  # First 10 tar files
        frames, clip_id = load_video_from_tar(tar_file, num_frames=8)

        if frames is None:
            continue

        frozen_psnr = eval_codec(frozen_codec, frames, "frozen",
                                save_images=True, output_dir=output_dir, clip_id=clip_id or tar_file.stem)
        trained_psnr = eval_codec(trained_codec, frames, "trained",
                                 save_images=True, output_dir=output_dir, clip_id=clip_id or tar_file.stem) if trained_codec else None

        frozen_psnrs.append(frozen_psnr)
        if trained_psnr:
            trained_psnrs.append(trained_psnr)

        diff = (trained_psnr - frozen_psnr) if trained_psnr else 0
        trained_str = f"{trained_psnr:.2f}" if trained_psnr else "N/A"
        diff_str = f"{diff:+.2f}" if trained_psnr else "N/A"

        print(f"{tar_file.stem[:24]:<24} {frozen_psnr:>13.2f} {trained_str:>14} {diff_str:>10}")

    print("="*70)

    if frozen_psnrs:
        print(f"\nMean PSNR:")
        print(f"  Frozen:  {sum(frozen_psnrs) / len(frozen_psnrs):.2f} dB")
        if trained_psnrs:
            print(f"  Trained: {sum(trained_psnrs) / len(trained_psnrs):.2f} dB")
            print(f"  Improvement: {sum(trained_psnrs) / len(trained_psnrs) - sum(frozen_psnrs) / len(frozen_psnrs):+.2f} dB")

    print(f"\n✓ Split-screen images saved to {output_dir}/")
    print("  Format: [LEFT: Input | RIGHT: Codec reconstruction]")

    return True

if __name__ == "__main__":
    sys.exit(0 if main() else 1)
