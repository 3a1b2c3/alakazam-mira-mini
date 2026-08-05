#!/usr/bin/env python3
"""Visualize codec reconstruction quality"""

import os
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["TORCH_COMPILE_DISABLE"] = "1"

import torch
import sys
from pathlib import Path
from PIL import Image
import math

try:
    from mira.codec.codec_model import VideoCodec
except ImportError:
    print("ERROR: mira not in PYTHONPATH")
    sys.exit(1)

def visualize_codec():
    codec_path = "outputs/codec_ckpt_7000.pth"
    if not Path(codec_path).exists():
        print(f"ERROR: {codec_path} not found")
        return False

    print(f"Loading codec from {codec_path}...")
    codec = VideoCodec.load_from_checkpoint(codec_path, device="cuda").eval()
    print("✓ Codec loaded")

    # Create test video (3 random frames)
    print("\nGenerating test video...")
    with torch.no_grad():
        # Random video [1, 3, 3, 256, 256] (batch, frames, channels, H, W)
        video = torch.randint(0, 256, (1, 3, 3, 256, 256), dtype=torch.float32, device="cuda") / 255.0

        # Encode
        input_norm, enc = codec.encode(video)
        print(f"  Input normalized: {input_norm.shape}")
        print(f"  Latent: {enc.z.shape}")

        # Decode
        recon = codec.decode(enc.z)
        print(f"  Reconstructed: {recon.shape}")

        # Align frames
        tt = min(input_norm.shape[1], recon.shape[1])
        input_aligned = input_norm[:, :tt]
        recon_aligned = recon[:, :tt]

        # Save first frame as image
        output_dir = Path("outputs/codec_viz")
        output_dir.mkdir(exist_ok=True)

        for i in range(tt):
            # Input frame (normalized to [-1, 1], convert to [0, 1])
            inp_frame = (input_aligned[0, i] * 0.5 + 0.5).clamp(0, 1).cpu()
            rec_frame = (recon_aligned[0, i] * 0.5 + 0.5).clamp(0, 1).cpu()

            # Convert to images
            inp_img = Image.fromarray((inp_frame.permute(1, 2, 0).numpy() * 255).astype('uint8'))
            rec_img = Image.fromarray((rec_frame.permute(1, 2, 0).numpy() * 255).astype('uint8'))

            # Save side-by-side
            combined = Image.new('RGB', (512, 256))
            combined.paste(inp_img, (0, 0))
            combined.paste(rec_img, (256, 0))
            combined.save(output_dir / f"frame_{i:03d}.png")
            print(f"  ✓ Saved frame_{i:03d}.png")

        # Compute PSNR
        mse = torch.nn.functional.mse_loss(recon_aligned, input_aligned).item()
        psnr = -10.0 * math.log10(mse + 1e-12)
        print(f"\n✓ PSNR: {psnr:.2f} dB")
        print(f"✓ Images saved to {output_dir}/")

    return True

if __name__ == "__main__":
    if visualize_codec():
        sys.exit(0)
    else:
        sys.exit(1)
