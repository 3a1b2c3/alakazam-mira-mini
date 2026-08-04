#!/usr/bin/env python3
"""Visualize codec reconstruction on real video data"""

import os
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["TORCH_COMPILE_DISABLE"] = "1"

import torch
import torch.nn.functional as F
import sys
import json
import math
import tempfile
import tarfile
from pathlib import Path
from PIL import Image
from einops import rearrange

try:
    from mira.codec.codec_model import VideoCodec
    from torchcodec.decoders import VideoDecoder
except ImportError:
    print("ERROR: mira/torchcodec not in PYTHONPATH")
    sys.exit(1)

def load_video_from_wds(shard_path: Path, match_id: str) -> torch.Tensor:
    """Load video clip from WebDataset tarfile"""
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
            frames = torch.stack([f.cpu() for f in dec])  # (T, C, H, W) uint8
        finally:
            try:
                Path(path).unlink()
            except OSError:
                pass

        return frames
    except Exception as e:
        print(f"  Error loading {match_id}: {e}")
        return None

def visualize_real_data():
    codec_path = "outputs/codec_ckpt_7000.pth"
    data_index = Path(r"C:\recordings\mira_wds\train\index.json")

    if not Path(codec_path).exists():
        print(f"ERROR: {codec_path} not found")
        return False

    if not data_index.exists():
        print(f"ERROR: {data_index} not found")
        return False

    print(f"Loading codec from {codec_path}...")
    codec = VideoCodec.load_from_checkpoint(codec_path, device="cuda").eval()
    print("✓ Codec loaded\n")

    # Load dataset index
    print(f"Loading data from {data_index}...")
    idx = json.load(open(data_index))
    root = data_index.parent
    entries = idx["entries"][:5]  # Test on first 5 clips
    print(f"✓ Loaded {len(entries)} entries\n")

    output_dir = Path("outputs/codec_viz_real")
    output_dir.mkdir(exist_ok=True)

    psnrs = []
    for i, e in enumerate(entries):
        shard = root / e["shard"]
        frames = load_video_from_wds(shard, e["match_id"])

        if frames is None:
            continue

        print(f"[{i+1}] {e['match_id'][:24]}")
        print(f"     Loaded {frames.shape[0]} frames, {frames.shape[1:]} shape")

        # Convert to tensor [0, 1] and add batch dim
        video = (frames.float() / 255.0).unsqueeze(0).to("cuda")  # (1, T, C, H, W)

        tgt_h, tgt_w = codec.config.encoder.video.height, codec.config.encoder.video.width

        with torch.no_grad():
            # Resize to codec input size (like codec_recon_psnr.py)
            b, t = video.shape[:2]
            video_resized = F.interpolate(
                rearrange(video, "b t c h w -> (b t) c h w"),
                size=(tgt_h, tgt_w), mode="bilinear", align_corners=False
            )
            video_resized = rearrange(video_resized, "(b t) c h w -> b t c h w", b=b, t=t)

            # Encode
            input_norm, enc = codec.encode(video_resized)
            recon = codec.decode(enc.z)

            # Align frames
            tt = min(input_norm.shape[1], recon.shape[1])
            input_aligned = input_norm[:, :tt]
            recon_aligned = recon[:, :tt]

            # Save first 3 frames
            for frame_idx in range(min(3, tt)):
                inp_frame = (input_aligned[0, frame_idx] * 0.5 + 0.5).clamp(0, 1).cpu()
                rec_frame = (recon_aligned[0, frame_idx] * 0.5 + 0.5).clamp(0, 1).cpu()

                inp_img = Image.fromarray((inp_frame.permute(1, 2, 0).numpy() * 255).astype('uint8'))
                rec_img = Image.fromarray((rec_frame.permute(1, 2, 0).numpy() * 255).astype('uint8'))

                combined = Image.new('RGB', (512, 256))
                combined.paste(inp_img, (0, 0))
                combined.paste(rec_img, (256, 0))
                combined.save(output_dir / f"clip_{i:02d}_frame_{frame_idx}.png")

            # Compute PSNR
            mse = torch.nn.functional.mse_loss(recon_aligned, input_aligned).item()
            psnr = -10.0 * math.log10(mse + 1e-12)
            psnrs.append(psnr)
            print(f"     PSNR: {psnr:.2f} dB")

    if psnrs:
        mean_psnr = sum(psnrs) / len(psnrs)
        print(f"\n✓ Mean PSNR: {mean_psnr:.2f} dB over {len(psnrs)} clips")
        print(f"✓ Images saved to {output_dir}/")
    return True

if __name__ == "__main__":
    if visualize_real_data():
        sys.exit(0)
    else:
        sys.exit(1)
