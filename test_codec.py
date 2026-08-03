#!/usr/bin/env python3
"""Test codec checkpoint-7000 encode/decode"""

import os
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["TORCH_COMPILE_DISABLE"] = "1"

import torch
import sys
from pathlib import Path

# Add mira to path if needed
try:
    from mira.codec.codec_model import VideoCodec
except ImportError:
    print("ERROR: mira not in PYTHONPATH. Run from mira venv or set PYTHONPATH")
    sys.exit(1)

def test_codec():
    codec_path = "outputs/codec_ckpt_7000.pth"

    if not Path(codec_path).exists():
        print(f"ERROR: {codec_path} not found")
        return False

    print(f"Loading codec from {codec_path}...")
    try:
        codec = VideoCodec.load_from_checkpoint(codec_path, device="cuda").eval()
        print("✓ Codec loaded")
    except Exception as e:
        print(f"✗ Failed to load: {e}")
        return False

    # Test with dummy video (1, 3, 3, 256, 256)
    print("\nTesting encode/decode...")
    try:
        with torch.no_grad():
            video = torch.randn(1, 3, 3, 256, 256, device="cuda")
            print(f"  Input: {video.shape}")

            input_norm, enc = codec.encode(video)
            print(f"  Encoded latent: {enc.z.shape}")

            recon = codec.decode(enc.z)
            print(f"  Reconstructed: {recon.shape}")

            # Align frame counts (codec may output different number of frames)
            tt = min(input_norm.shape[1], recon.shape[1])
            input_aligned = input_norm[:, :tt]
            recon_aligned = recon[:, :tt]

            # Compute MSE
            mse = torch.nn.functional.mse_loss(recon_aligned, input_aligned).item()
            psnr = -10.0 * torch.log10(torch.tensor(mse)).item()
            print(f"\n✓ MSE: {mse:.6f}, PSNR: {psnr:.2f} dB (aligned to {tt} frames)")
    except Exception as e:
        print(f"✗ Encode/decode failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True

if __name__ == "__main__":
    if test_codec():
        print("\n✓ Codec test passed")
        sys.exit(0)
    else:
        print("\n✗ Codec test failed")
        sys.exit(1)
