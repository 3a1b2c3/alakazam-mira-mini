# Generate Videos from World-Model Checkpoints

Generate inference videos from trained world-model checkpoints on RacerX test data.

## Usage

### Default (checkpoint + auto-detect video)
```powershell
.\generate_video_racerx.bat
```
Uses default checkpoint (`wm_ckpt_49000.pth`) and finds first available video from:
1. `C:\recordings\mira_wds\test\000\` (RacerX test data)
2. `data\` (local fallback)

### Custom Checkpoint
```powershell
.\generate_video_racerx.bat outputs\wm_ckpt_98000.pth
.\generate_video_racerx.bat outputs\wm_ckpt_70000.pth
```

### Custom Checkpoint + Custom Video
```powershell
.\generate_video_racerx.bat outputs\wm_ckpt_98000.pth data\1ea1a4ce-2fdc-44fe-afdd-8019fbacea28_clip00000_c00003.mp4
.\generate_video_racerx.bat outputs\wm_ckpt_70000.pth C:\recordings\mira_wds\test\000\clip001.mp4
```

## What It Does

1. **Checkpoint**: Accepts checkpoint path as arg 1 (default: `wm_ckpt_49000.pth`)
2. **Video**: Accepts video path as arg 2 (auto-detects if not provided)
3. **Validation**: Checks both checkpoint and video exist
4. **Video Search Order**:
   - Arg 2 (if provided)
   - `C:\recordings\mira_wds\test\000\` (RacerX test data)
   - `data\` (local test directory)
5. **Inference**: `python generate_from_inference.py <checkpoint> --video <video_path>`
6. **Output**: `outputs/generated_native_wm_ckpt_<STEP>_c<CLIP_ID>.mp4`

## Available Checkpoints

```
outputs\wm_ckpt_49000.pth    (baseline WM, 49k steps)
outputs\wm_ckpt_70000.pth    (converged WM, 70k steps)
outputs\wm_ckpt_98000.pth    (best WM, 98k steps, 17.87 dB PSNR)
outputs\codec_ckpt_33000.pth (best codec, 24.60 dB PSNR on RacerX)
```

## Examples

```powershell
cd C:\workspace\world\alakazam-mira-mini

# Generate with best WM checkpoint (use generate_video_with_actions.bat)
.\generate_video_with_actions.bat outputs\wm_ckpt_98000.pth

# Generate with best codec checkpoint
.\generate_video_with_actions.bat outputs\codec_ckpt_33000.pth

# List all available checkpoints
dir /b outputs\*.pth
```

## Output

Generated video saved to: `outputs\generated_native_wm_ckpt_<STEP>_c<CLIP_ID>.mp4`

Example: `generated_native_wm_ckpt_98000_c00001.mp4`

## Troubleshooting

**Error: Checkpoint not found**
```
Solution: Specify full path to checkpoint
.\generate_video_racerx.bat outputs\wm_ckpt_98000.pth
```

**Error: No RacerX videos found**
```
Solution: Ensure C:\recordings\mira_wds\test\000\ contains .mp4 files
```

## Related Scripts

- `eval_video_quality.py` — Evaluate generated video quality
- `eval_codec_comparisons.py` — Compare codec outputs
- `eval_fvd.py` — Compute FVD metric
