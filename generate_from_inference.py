#!/usr/bin/env python3
"""Generate video using mira inference API with proper VideoActionBatch"""

import os
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["TORCH_COMPILE_DISABLE"] = "1"

import torch
import sys
import argparse
from pathlib import Path
import cv2
import numpy as np

try:
    from mira.inference.loading import load_world_model
    from mira.data.batch import VideoActionBatch
    from mira.world_model.actions_config import ActionTensors, ActionConfig
    from mira.world_model.config import WorldModelInferenceConfig
    from torchcodec.decoders import VideoDecoder
except ImportError as e:
    print(f"ERROR: {e}")
    sys.exit(1)

def load_context_video(video_path: str, num_frames: int = 8):
    """Load context frames from video"""
    path = Path(video_path)
    if not path.exists():
        print(f"ERROR: {path} not found")
        return None

    try:
        dec = VideoDecoder(str(path))
        frames = torch.stack([f.cpu() for f in dec])[:num_frames]
        return frames.float() / 255.0  # Normalize to [0, 1]
    except Exception as e:
        print(f"ERROR: Failed to load video: {e}")
        return None

def load_actions_from_jsonl(video_path: str, num_frames: int = None):
    """Load actions from .p0.jsonl file corresponding to video"""
    import json

    video_path = Path(video_path)
    action_path = video_path.parent / f"{video_path.stem}.jsonl"

    if not action_path.exists():
        print(f"  WARNING: Action file not found: {action_path}")
        return None

    try:
        actions_list = []
        with open(action_path, 'r') as f:
            for line in f:
                if line.strip():
                    actions_list.append(json.loads(line))
                    if num_frames and len(actions_list) >= num_frames:
                        break

        print(f"  Loaded {len(actions_list)} action frames from {action_path.name}")
        return actions_list
    except Exception as e:
        print(f"  ERROR loading actions: {e}")
        return None

def actions_list_to_tensors(actions_list, action_cfg, batch_size: int = 1):
    """Convert list of action dicts to ActionTensors"""
    if not actions_list:
        return None

    num_frames = len(actions_list)
    num_keys = len(action_cfg.valid_keys)

    # Create key_presses tensor (multi-hot encoding)
    key_presses = torch.zeros((batch_size, num_frames, num_keys), dtype=torch.int32)

    # Convert each action frame
    for t, action_dict in enumerate(actions_list):
        if 'keys' in action_dict:
            pressed_keys = action_dict['keys']
            if isinstance(pressed_keys, list):
                for key_name in pressed_keys:
                    if key_name in action_cfg.valid_keys:
                        key_idx = action_cfg.valid_keys.index(key_name)
                        key_presses[0, t, key_idx] = 1

    # Create mouse movements (zero for now, as jsonl only has keys)
    mouse_movements = torch.zeros((batch_size, num_frames, 2), dtype=torch.float32)

    # Create ActionTensors object
    actions = ActionTensors(config=action_cfg, batch_size=batch_size)
    actions.key_presses = key_presses
    actions.mouse_movements = mouse_movements
    actions.game_mouse_sensitivity = torch.full((batch_size,), float('nan'), dtype=torch.float32)

    return actions

def main():
    ap = argparse.ArgumentParser(description="Generate video using mira inference")
    ap.add_argument("checkpoint", nargs="?", default="outputs/wm_ckpt_49000.pth",
                    help="world model checkpoint")
    ap.add_argument("--video", default=r"C:\workspace\world\mira\dataset_00000\2026-05-08T20-37-10Z-5f298d_c00000.p0.mp4",
                    help="context video")
    ap.add_argument("--output", help="output MP4 (default: auto-generated)")
    ap.add_argument("--context-frames", type=int, default=8, help="context frames")
    ap.add_argument("--gen-frames", type=int, default=16, help="frames to generate")
    args = ap.parse_args()

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"ERROR: {ckpt_path} not found")
        return False

    print("="*70)
    print("Generate Video (Mira Inference API)")
    print("="*70)

    # Load model
    print(f"\nLoading checkpoint: {ckpt_path.name}")
    try:
        model, run_cfg = load_world_model(str(ckpt_path), device="cuda")
        print("✓ Model loaded")
    except Exception as e:
        print(f"ERROR: Failed to load: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Load context video
    print(f"\nLoading context video: {Path(args.video).name}")
    context = load_context_video(args.video, args.context_frames)
    if context is None:
        return False
    print(f"✓ Context shape: {context.shape}")

    # Load actions from JSONL
    print(f"\nLoading actions...")
    actions_list = load_actions_from_jsonl(args.video, num_frames=args.context_frames + args.gen_frames)

    # Generate frames using model
    print(f"\nGenerating {args.gen_frames} frames...")
    output_path = args.output or f"outputs/generated_{ckpt_path.stem}_api.mp4"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    try:
        # Don't create writer yet - will use native model resolution
        # (determined after inference)
        h, w = context.shape[2:4]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = None

        frames_written = 0

        with torch.no_grad():

            # Generate future frames using model.inference()
            print(f"  Running model.inference()...")

            try:
                # Prepare context: (T, C, H, W) [0, 1] -> (B, T, C, H, W) uint8
                context_uint8 = (context * 255).byte()  # (context_frames, C, H, W) uint8
                # inference() generates for the WHOLE video's latents (keeping the
                # first n_context_latents as real context), so the video must span
                # context + gen frames. Pad with the last context frame; those
                # placeholder frames get replaced by noise and denoised.
                pad = context_uint8[-1:].repeat(args.gen_frames, 1, 1, 1)
                full_video = torch.cat([context_uint8, pad], dim=0)  # (context+gen, C, H, W)
                context_batch_video = full_video.unsqueeze(0).to("cuda")  # (1, context+gen, 3, H, W)

                # Get action config from model
                if hasattr(model, 'config') and hasattr(model.config, 'action'):
                    action_cfg = model.config.action
                else:
                    action_cfg = ActionConfig(valid_keys=['w', 'a', 's', 'd'], source_fps=20, target_fps=10)

                # Use loaded actions if available, else create neutral actions
                if actions_list and len(actions_list) >= (args.context_frames + args.gen_frames):
                    actions = actions_list_to_tensors(actions_list, action_cfg, batch_size=1)
                    print(f"  Using loaded actions from JSONL")
                else:
                    # Fallback: create neutral actions
                    print(f"  Using neutral (zero) actions")
                    actions = ActionTensors(config=action_cfg, batch_size=1)
                    total_frames = args.context_frames + args.gen_frames
                    dummy_keys = torch.zeros((1, total_frames, len(action_cfg.valid_keys)), dtype=torch.int32)
                    dummy_mouse = torch.zeros((1, total_frames, 2), dtype=torch.float32)
                    actions.key_presses = dummy_keys
                    actions.mouse_movements = dummy_mouse
                    actions.game_mouse_sensitivity = torch.full((1,), float('nan'), dtype=torch.float32)

                # Create VideoActionBatch with context + space for generation
                batch = VideoActionBatch(video=context_batch_video, actions=actions)
                batch = batch.to("cuda")

                # Call model.inference() which handles everything
                print(f"  Running inference ({args.gen_frames} frames)...")
                # Create inference config with proper denoising steps
                inference_config = WorldModelInferenceConfig(
                    n_diffusion_steps=10,
                    noise_level=0.2,
                    schedule_type='linear_quadratic'
                )
                # inference() takes (batch, config, progress_bar) and returns an
                # InferenceOutputs; frame count is driven by the batch video length
                # (context + gen, padded above), not a kwarg.
                gen_rgb = model.inference(batch=batch, config=inference_config, progress_bar=True).output_video  # (1, T, 3, H, W)

                # Convert to [0, 255] uint8
                if gen_rgb.max() <= 1.0:
                    gen_frames = (gen_rgb * 255).byte()
                else:
                    gen_frames = gen_rgb.byte()

                gen_h, gen_w = gen_frames.shape[3:5]  # (B, T, C, H, W) -> H, W
                print(f"  Generated video shape: {tuple(gen_frames.shape)}")
                print(f"  Using native model resolution: {gen_w}x{gen_h}")

                # Create writer with native resolution
                writer = cv2.VideoWriter(output_path, fourcc, 20, (gen_w, gen_h))
                frames_written = 0

                # Write ALL frames at native resolution (no upscaling)
                for t in range(gen_frames.shape[1]):
                    frame = gen_frames[0, t]  # (3, H, W) uint8
                    frame_np = frame.cpu().numpy().transpose(1, 2, 0)  # (H, W, 3) uint8
                    frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
                    writer.write(np.ascontiguousarray(frame_bgr))
                    frames_written += 1

                    if t < args.context_frames:
                        label = f"context {t}"
                    else:
                        label = f"generated {t - args.context_frames}"

                    if t % max(1, gen_frames.shape[1] // 4) == 0:
                        print(f"  Wrote {label}")

            except Exception as e:
                print(f"\nERROR: Model inference failed")
                import traceback
                traceback.print_exc()
                writer.release()
                return False

        writer.release()
        print(f"\n✓ Video saved: {Path(output_path).resolve()}")
        print(f"Total frames: {frames_written}")
        return True

    except Exception as e:
        print(f"ERROR: Failed to generate: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    sys.exit(0 if main() else 1)
