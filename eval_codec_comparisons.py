#!/usr/bin/env python3
"""Compute PSNR from codec comparison directories"""

import json
import math
from pathlib import Path
import numpy as np
from PIL import Image

def compute_psnr_from_images(dir_path, pattern_frozen="_frozen_frame_", pattern_trained="_trained_frame_"):
    """Compute PSNR between frozen and trained codec frames"""
    dir_path = Path(dir_path)
    if not dir_path.exists():
        print(f"ERROR: {dir_path} not found")
        return None

    frozen_files = sorted(dir_path.glob(f"*{pattern_frozen}*.png"))
    trained_files = sorted(dir_path.glob(f"*{pattern_trained}*.png"))

    if not frozen_files or not trained_files:
        print(f"ERROR: No image files found in {dir_path}")
        return None

    psnr_values = []
    for frozen_path, trained_path in zip(frozen_files, trained_files):
        try:
            frozen = np.array(Image.open(frozen_path)).astype(np.float32) / 255.0
            trained = np.array(Image.open(trained_path)).astype(np.float32) / 255.0

            mse = np.mean((frozen - trained) ** 2)
            if mse == 0:
                psnr = 100.0
            else:
                psnr = -10.0 * math.log10(mse)
            psnr_values.append(psnr)
        except Exception as e:
            print(f"  Error processing {frozen_path.name}: {e}")
            continue

    if psnr_values:
        return {
            'mean_psnr': np.mean(psnr_values),
            'std_psnr': np.std(psnr_values),
            'min_psnr': np.min(psnr_values),
            'max_psnr': np.max(psnr_values),
            'num_frames': len(psnr_values),
        }
    return None

def main():
    outputs_dir = Path("outputs")
    comparisons = [
        ("shard_000_frozen_vs_codec_ckpt_15000_n400", "codec_ckpt_15000"),
        ("shard_000_frozen_vs_codec_ckpt_25000_n200", "codec_ckpt_25000 (n=200)"),
        ("shard_000_frozen_vs_codec_ckpt_25000_n400", "codec_ckpt_25000 (n=400)"),
        ("shard_000_frozen_vs_trained_7000", "codec_ckpt_7000"),
    ]

    print("="*70)
    print("CODEC COMPARISON: PSNR (Frozen vs Trained)")
    print("="*70)
    print(f"{'Comparison':<35} {'PSNR Mean':<12} {'Std Dev':<12} {'Frames':<8}")
    print("-"*70)

    results = {}
    for dir_name, label in comparisons:
        dir_path = outputs_dir / dir_name
        if dir_path.exists():
            result = compute_psnr_from_images(dir_path)
            if result:
                results[label] = result
                print(f"{label:<35} {result['mean_psnr']:<12.2f} {result['std_psnr']:<12.2f} {result['num_frames']:<8}")
            else:
                print(f"{label:<35} ERROR")
        else:
            print(f"{label:<35} Not found")

    print("="*70)
    print("\nINTERPRETATION:")
    print("- Higher PSNR = better codec (more similar to frozen)")
    print("- PSNR > 30 dB = very good (imperceptible diff)")
    print("- PSNR 20-30 dB = acceptable quality")
    print("- PSNR < 20 dB = visible degradation")

    if results:
        best = max(results.items(), key=lambda x: x[1]['mean_psnr'])
        print(f"\n✓ Best codec: {best[0]} ({best[1]['mean_psnr']:.2f} dB)")

    return True

if __name__ == "__main__":
    import sys
    sys.exit(0 if main() else 1)
