# MIRA-Mini Training & Testing Skills

Quick reference for all scripts and workflows.

---

## 1. Codec Training (Stage 1)

### Start Codec Training
```bash
# Linux/WSL
./train_codec.sh

# Windows (via WSL)
wsl bash -c "cd /mnt/c/workspace/world/alakazam-mira-mini && ./train_codec.sh"
```

**Outputs**: `codec_logs/checkpoint-*/checkpoint.pth`

**Monitor**: Background PSNR evaluation in `codec_psnr_log.txt`

**Auto-resume**: Continues from latest checkpoint

---

## 2. World Model Training (Stage 2)

### Train on Custom Codec
```bash
# Linux/WSL - auto-detects codec from codec_logs/
./train.sh

# Windows Stage 1 - isolated output dir
train_stage1.bat

# With custom checkpoint
./train.sh run.steps=50000
```

**Codec priority**:
1. Custom: `codec_logs/checkpoint-*/checkpoint.pth`
2. Frozen fallback: mira-mini from HF

---

## 3. Codec Quality Testing

### Test Reconstruction PSNR
```bash
# PowerShell
cd C:\workspace\world\alakazam-mira-mini
python codec_recon_psnr.py `
  --checkpoint checkpoint-10000-codec-horde.pth `
  --num-samples 16

# Output: PSNR stats, per-frame quality, temporal consistency
```

**What it tests**: Encode→decode ceiling (world model can't beat this)

---

## 4. MIND Benchmark Scoring

### Score World Model on MIND
```bash
cd C:\workspace\world\MIND

# Score with all metrics (including GSC)
./run_mind.bat lingbot-v2 lcm,visual,dino,action,gsc 1 both

# Or just GSC
./run_mind.bat lingbot-v2 gsc 1 both
```

**150 lingbot-v2 samples**:
- 1st person: GSC avg_mse = 0.0750
- 3rd person: GSC avg_mse = 0.0331 (better)

**Results**: `result_lingbot-v2_*.json`

---

## 5. OmniDreams Performance Testing

### Native UI with Performance Optimizations
```bash
cd C:\workspace\world\flashdream_public

# One-time setup
setup_interactive_native.bat

# Run performance mode
run_interactive_drive_perf.bat
```

**Features**:
- CUDA graphs + torch.compile + FP8
- Native Windows UI (no WebRTC)
- 1168×640 @ 30 FPS

**Monitor**: nvidia-smi for power/util/memory

---

## 6. Remote Checkpoint Management

### Copy from Horde Server
```bash
# From local PowerShell
scp horde@10.57.233.223:/home/horde/mira/codec_logs/checkpoint-10000/checkpoint.pth `
  "C:\workspace\world\alakazam-mira-mini\checkpoint-10000-codec-horde.pth"

scp horde@10.57.233.223:/home/horde/mira/train_world_model_logs_scratch/checkpoint-21000/checkpoint.pth `
  "C:\workspace\world\alakazam-mira-mini\checkpoint-21000-wm-horde.pth"
```

### List Available Checkpoints
```bash
# SSH into horde
ssh horde@10.57.233.223

# List codec checkpoints
ls -lh /home/horde/mira/codec_logs/checkpoint-*/checkpoint.pth | tail -5

# List world model checkpoints
ls -lh /home/horde/mira/train_world_model_logs_scratch/checkpoint-*/checkpoint.pth | tail -5
```

---

## 7. Benchmarking & Profiling

### Ludus Renderer Benchmarks
```bash
run_perf_benchmarks.bat
```

**Benchmarks**:
- Single-scene rendering
- Multi-camera performance
- GPU↔CPU transfer latency
- Batch scaling behavior

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| DINO weights missing | `./download_dino.sh` |
| GPU OOM during training | Reduce batch_size, enable expandable_segments |
| CUDA graphs crash (Blackwell) | Fallback to eager: `run_interactive_drive.bat` |
| GSC not in MIND results | Verify process.py has line 298 fix |
| SSH auth fails (publickey) | Ensure SSH key in `~/.ssh/id_rsa` |
| Codec test won't run | Check mira venv has torchcodec installed |

---

## Key Metrics & Targets

| Metric | Direction | Typical Range | Benchmark Leader |
|--------|-----------|---------------|------------------|
| LCM PSNR | ↑ | 8–50 dB | HY-WorldPlay (15.12) |
| GSC avg_mse | ↓ | 0–1 | ABot (0.0201) |
| Visual quality | ↑ | 0–1 | Yume (0.577) |
| DINO consistency | ↓ | 0–0.01 | HY-WorldPlay (0.000334) |
| Action accuracy | ↓ | 0–1 | Yume (0.008) |

---

## File Locations

```
C:\workspace\world\
├── alakazam-mira-mini/          ← Main project
│   ├── train.sh / train_stage1.bat
│   ├── train_codec.sh
│   ├── codec_recon_psnr.py
│   ├── checkpoint-*-horde.pth   ← Downloaded checkpoints
│   └── codec_logs/              ← Training outputs
├── mira/                         ← MIRA trainer (sibling)
├── MIND/                         ← Benchmark suite
│   └── result_lingbot-v2_*.json
└── flashdream_public/
    └── run_interactive_drive_perf.bat
```

---

## Recent Updates

- **process.py (MIND)**: GSC now calculates for all test types, handles odd frames
- **train_codec.sh**: Added background PSNR monitor + auto-resume
- **train.sh**: Custom codec detection (codec_logs/ priority)
- **Documentation**: Full skills guide with all command examples
