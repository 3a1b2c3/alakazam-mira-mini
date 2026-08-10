#!/usr/bin/env python3
"""Create side-by-side splitscreen video (generated vs ground truth)"""

import cv2
from pathlib import Path

def create_splitscreen(real_path, gen_path, output_path):
    """Create side-by-side comparison"""
    print(f"Creating splitscreen for {Path(gen_path).name}...")

    real_cap = cv2.VideoCapture(real_path)
    gen_cap = cv2.VideoCapture(gen_path)

    # Get properties
    fps = gen_cap.get(cv2.CAP_PROP_FPS)
    gen_w = int(gen_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    gen_h = int(gen_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    real_w = int(real_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    real_h = int(real_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Output: side by side
    out_h = max(gen_h, real_h)
    out_w = gen_w + real_w + 10  # 10px gap
    gap = 10

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (out_w, out_h))

    frame_count = 0
    while True:
        ret_real, real_frame = real_cap.read()
        ret_gen, gen_frame = gen_cap.read()

        if not ret_real or not ret_gen:
            break

        # Resize to consistent height
        real_frame = cv2.resize(real_frame, (real_w, out_h))
        gen_frame = cv2.resize(gen_frame, (gen_w, out_h))

        # Create combined frame
        combined = cv2.hconcat([real_frame, cv2.cvtColor(cv2.cvtColor(gen_frame, cv2.COLOR_RGB2BGR), cv2.COLOR_BGR2RGB)])
        combined = cv2.copyMakeBorder(combined, 0, 0, 0, gap, cv2.BORDER_CONSTANT, value=(50, 50, 50))

        # Add text labels
        cv2.putText(combined, "Ground Truth", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        cv2.putText(combined, "Generated (119000)", (gen_w + gap + 10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)

        # Resize to output size if needed
        combined = cv2.resize(combined, (out_w, out_h))
        writer.write(combined)
        frame_count += 1

    writer.release()
    real_cap.release()
    gen_cap.release()

    print(f"  ✓ {frame_count} frames | Saved: {output_path}")

if __name__ == "__main__":
    real = "C:/recordings/mira_wds/test/000/dataset_00000/1ea1a4ce-2fdc-44fe-afdd-8019fbacea28_clip00000_c00000.p0.mp4"
    gen = "outputs/demo_119000_c00000.mp4"
    out = "outputs/splitscreen_119000_c00000.mp4"

    create_splitscreen(real, gen, out)
    print(f"\n✅ Full path: {Path(out).absolute()}")
