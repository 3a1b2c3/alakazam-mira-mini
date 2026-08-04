# MIRA-Mini Scripts Guide

## Overview
Scripts for training MIRA codec and latent world model from scratch with custom codecs.

---

## Core Training Scripts

### `train_codec.sh` (Linux/WSL)
**Train RAEv2 temporal video codec from scratch**

```bash
./train_codec.sh                          # Default: codec_logs/
./train_codec.sh run.steps=125000         # Custom steps
./train_codec.sh run.batch_size=2         # Custom batch
```

**Output**: `codec_logs/checkpoint-*/checkpoint.pth`

**Features**:
- No frozen codec dependency
- Uses DINO for latent consistency
- Auto-resumes from latest checkpoint
- Batch processing with distributed training support

**Environment**:
- `RUN`: env prefix (default: `pixi run`)
- `TRAIN_INDEX` / `TEST_INDEX`: data indices
- `WORKERS`: dataloader workers (default: 4)

---

### `train.sh` (Linux/WSL)
**Train latent world model on custom codec**

```bash
./train.sh                               # Auto-detects codec from codec_logs/
./train.sh run.steps=50000               # Custom steps
./train.sh model/latent_world_model=1b   # Full 1B model
```

**Codec Priority**:
1. Custom trained codec: `codec_logs/checkpoint-*/checkpoint.pth`
2. Fallback: Frozen mira-mini codec from HF

**Features**:
- Auto-resumes from latest checkpoint
- DINO metrics evaluation
- Offline W&B logging
- CUDA memory fragmentation reduction

---

### `train_stage1.bat` (Windows)
**Windows batch script for stage 1 training**

```batch
train_stage1.bat                    # Default: stage1_logs/
train_stage1.bat run.steps=10000    # Custom steps
train_stage1.bat model/latent_world_model=1b
```

**Features**:
- No auto-resume (trains from scratch)
- Output dir: `stage1_logs/` (isolated)
- Single-player, batch=1, no compile
- HF_TOKEN required (from `.env` or shell)

**Differences from train.sh**:
- Always starts fresh (no auto-resume)
- Separate output directory
- Designed for initial stage 1 runs

---

## Performance & Evaluation

### `codec_recon_psnr.py` (Linux/WSL/Windows)
**Test codec reconstruction quality on real data**

```bash
cd C:\workspace\world\alakazam-mira-mini

# Test with horde checkpoint
..\mira\.venv\Scripts\python.exe codec_recon_psnr.py \
  --checkpoint checkpoint-10000-codec-horde.pth \
  --num-samples 16

# Or with dataset index
..\mira\.venv\Scripts\python.exe codec_recon_psnr.py \
  --checkpoint checkpoint-10000-codec-horde.pth \
  --data C:\recordings\mira_wds\train\index.json \
  --num-samples 16
```

**What it measures**:
- Encode→decode reconstruction PSNR
- Codec quality ceiling (world model can't beat this)
- Per-frame PSNR and temporal consistency
- Identifies domain-transfer bottlenecks

**Output**: `codec_test_results.txt` with PSNR stats

**Requirements**:
- mira venv with torchcodec
- TORCHDYNAMO_DISABLE auto-set (eager mode)
- DINO cached (no GitHub access needed)

---

### `run_perf_benchmarks.bat` (Windows)
**Comprehensive Ludus renderer performance benchmarks**

```batch
run_perf_benchmarks.bat
```

**Benchmarks**:
1. Single-scene (10 iterations)
2. Multi-camera rendering
3. GPU interop (transfer latency)

**Metrics**:
- Render time, FPS, GPU memory
- Load/upload times
- Batch scaling behavior

---

## MIND Benchmark Scoring

### MIND GSC Calculation (Fixed)
**Location**: `C:\workspace\world\MIND\src\process.py`

**Issue**: GSC was only calculated for mirror_test, not mem_test/action_space_test

**Fix Applied**:
- Added GSC to condition for non-mirror_test types (line 298)
- Handle odd frame counts (drop last frame for even split)
- Full VideoStreamReader read before mirror split

**Usage**:
```bash
cd C:\workspace\world\MIND
./run_mind.bat lingbot-v2 gsc 1 both
```

**Results** (150 lingbot-v2 samples):
- 1st person: GSC avg_mse = 0.0750 ± 0.0455
- 3rd person: GSC avg_mse = 0.0331 ± 0.0308 (better)

---

## OmniDreams Performance Mode

### Native UI (Performance Mode)
**Location**: `C:\workspace\world\flashdream_public\`

```bash
run_interactive_drive_perf.bat           # Native UI + CUDA graphs + torch.compile + FP8
```

**Prerequisites**:
```bash
setup_interactive_native.bat              # One-time setup
```

**Features**:
- CUDA graphs for kernel fusion
- torch.compile JIT optimization
- FP8 quantization
- 1168×640 @ 30 FPS
- Local attention (size 6)

**Config**: `omnidreams\interactive_drive\configs\example_world_model_perf.yaml`

---

## Configuration Files

### `world_model_config.yaml` (OmniDreams)
**Performance-tuned world model manifest**

Key settings:
- `resolution_wh`: [1168, 640]
- `fps`: 30
- `compile_net`: true
- `native_dit_acceleration`: required
- `native_dit_backend`: fp8_kvcache_cudnn
- `denoising_steps`: [1000, 100]

---

## Environment Setup

### Required
```bash
# HuggingFace token
export HF_TOKEN="your_token_here"

# DINO weights (for MIRA training)
./download_dino.sh

# Dataset
./get_data.sh                # Downloads WebDataset, writes data_paths.sh
```

### Optional (Performance)
```bash
# GPU memory optimization
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# CUDA fragmentation
export TORCHINDUCTOR_COMPILE_THREADS=1
```

---

## Troubleshooting

### MIRA Training
- **ModuleNotFoundError**: Run `pixi sync` in mira directory
- **DINO weights missing**: Run `./download_dino.sh`
- **GPU OOM**: Reduce `batch_size`, enable `expandable_segments`

### OmniDreams Performance
- **CUDA driver error (Blackwell)**: Fallback to eager mode: `run_interactive_drive.bat`
- **Disk space**: Move HF_HOME to larger drive
- **torch.compile fails**: Set `TORCHDYNAMO_DISABLE=1`

### MIND GSC Scoring
- **GSC not in results**: Ensure process.py has line 298 fix
- **Odd frame counts**: Auto-handled (drops last frame)
- **No test data**: Populate `MIND-tests/lingbot-v2/{1st,3rd}_data/{action_space_test,mem_test}/`

---

## Key Metrics Reference

| Metric | Direction | Range | Notes |
|--------|-----------|-------|-------|
| LCM PSNR | ↑ Higher | 8–50 dB | Reconstruction quality |
| GSC avg_mse | ↓ Lower | 0–1 | Memory consistency |
| Visual | ↑ Higher | 0–1 | Perceptual quality |
| DINO mse | ↓ Lower | 0–0.01 | Feature space consistency |
| Action avg_mse | ↓ Lower | 0–1 | Control accuracy |

---

## File Locations

```
C:\workspace\world\
├── alakazam-mira-mini/
│   ├── train.sh / train_stage1.bat         ← World model training
│   ├── train_codec.sh                      ← Codec training
│   ├── codec_logs/                         ← Codec checkpoints
│   └── outputs/                            ← World model outputs
├── mira/                                   ← MIRA trainer (sibling)
├── MIND/                                   ← MIND benchmark
│   ├── src/process.py                      ← GSC fix applied here
│   └── result_lingbot-v2_*.json            ← Benchmark results
└── flashdream_public/
    └── run_interactive_drive_perf.bat      ← Native UI performance mode
```

---

## Recent Changes

**train_codec.sh**: Added background PSNR monitor + auto-resume
**train.sh**: Custom codec detection (codec_logs/ priority) + frozen fallback
**process.py** (MIND): GSC now calculated for all test types, handles odd frames
