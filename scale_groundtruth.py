#!/usr/bin/env python3
"""Scale ground truth video to match generated resolution"""

import cv2
from pathlib import Path

input_path = "outputs/groundtruth_1ea1a4ce_c00000.mp4"
output_path = "outputs/groundtruth_1ea1a4ce_c00000_288x512.mp4"

# Target resolution
target_w, target_h = 512, 288

print(f"Scaling {input_path} to {target_w}x{target_h}...")

# Read video
cap = cv2.VideoCapture(input_path)
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"Input: {total_frames} frames @ {fps} fps")

# Create writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
writer = cv2.VideoWriter(output_path, fourcc, fps, (target_w, target_h))

frames_written = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize frame
    resized = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
    writer.write(resized)
    frames_written += 1

    if frames_written % 8 == 0:
        print(f"  Scaled frame {frames_written}/{total_frames}")

cap.release()
writer.release()

print(f"\nScaled video saved: {Path(output_path).resolve()}")
print(f"Resolution: {target_w}x{target_h}")
print(f"Frames: {frames_written}")
