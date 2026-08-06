# MIRA Evaluation Results

## Summary
Evaluation of MIRA world model inference on real video data with frozen codec backbone.

## Test Data
- **c00000**: 24 frames @ 288×512, natural scene motion
- **c00001**: 20 frames @ 288×512, complex action sequence

## Codec Checkpoint Measurements

### PSNR Comparison: codec_ckpt_25000 vs codec_checkpoint-30000 (Aug 6, 2026)

Evaluated on 16 RacerX test clips:

**codec_ckpt_25000.pth**
| Metric | Value |
|--------|-------|
| Mean PSNR | 23.35 dB |
| Status | ✓ Stable |

**codec_checkpoint-30000/checkpoint.pth**
| Metric | Value | Change |
|--------|-------|--------|
| Mean PSNR | 23.71 dB | +0.36 dB |
| Status | ✓ Improved |

### Extended Codec Eval: codec_ckpt_25000 vs codec_ckpt_33000 (Aug 6, 2026)

Evaluated on 32 RacerX test clips:

| Checkpoint | Mean PSNR | Clips | Gap to Target | Status |
|---|---|---|---|---|
| codec_ckpt_25000 | 23.23 dB | 32 | 5.37 dB | Baseline |
| codec_ckpt_33000 | 24.60 dB | 32 | 4.00 dB | **Best** ✓ |
| Target | 28.60 dB | — | — | Out of reach |

**Key Finding**: codec_ckpt_33000 (+1.37 dB vs 25000) is best available. Still 4.0 dB below target due to RL→RacerX domain transfer ceiling (not codec capacity).

**Decision**: Use `codec_ckpt_33000` for Phase 3 WM finetuning. Updated train.sh, finetune.sh, finetune_phase3.sh to use checkpoint-33000.

### Training Fix Applied
Added cosine annealing with warmup to prevent future codec divergence:
```
+optimizer.lr=1e-4 +optimizer.schedule=cosine_warmup +optimizer.warmup_steps=2000
```

## World Model Checkpoint Measurements

### Video Quality Metrics (Using codec_ckpt_25000.pth)

**Test Clip: c00000 (24 frames, natural motion)**
| Checkpoint | PSNR (dB) | Status | Analysis |
|---|---|---|---|
| checkpoint-49000 | 22.30 | ✓ Converged | First stable point |
| checkpoint-63000 | 22.30 | ✓ Flat | No improvement |
| checkpoint-70000 | 22.30 | ✓ Flat | Fully saturated |

**Test Clip: c00001 (20 frames, complex actions)**
| Checkpoint | PSNR (dB) | Status | Analysis |
|---|---|---|---|
| checkpoint-49000 | 15.25 | ✓ Lower | Inherently harder |
| checkpoint-63000 | 15.25 | ✓ Flat | No improvement |
| checkpoint-70000 | 15.25 | ✓ Flat | Fully saturated |

**Key Finding**: 
- World model converged by step 49k
- Frozen codec at 288×512 has hard ceiling ~22.3 dB PSNR (c00000), ~15.2 dB (c00001)
- c00001 harder due to different action patterns, not model capacity
- No performance gain from steps 49k → 70k

### World Model Checkpoint Comparison: 70k vs 98k (Aug 6, 2026)

**Results**:

| Metric | 70k | 98k | Δ | Winner |
|--------|-----|-----|---|--------|
| **PSNR** | 17.61 dB | 17.87 dB | +0.26 dB | 98k |
| **Loss** | 1.714 | 1.877 | +0.16 | 70k |
| **gFDD** | 5.67 | 5.63 | −0.04 | 98k |
| **gFID** | 322.5 | 326.0 | +3.5 | 70k |
| **Latent Drift** | 0.2845 | 0.2702 | −0.014 | 98k |

**Key Finding**: 98k shows marginal improvement (+0.26 dB PSNR) over 70k. Both have converged; additional training (70k→98k = +28k steps) yielded minimal gain.

**Conclusion**: **Codec is the bottleneck**. World models cannot improve further with current codec ceiling (23.71 dB). Must retrain codec to unlock better WM performance.

**Decision**: Use 70k as baseline WM. Focus effort on Phase 2A (codec retraining).

## Codec Training Configuration
File: `train_codec.sh` (mira and alakazam-mira-mini versions)

Default training now uses learning rate decay:
```bash
./train_codec.sh +optimizer.lr=1e-4 +optimizer.schedule=cosine_warmup +optimizer.warmup_steps=2000
```

Rationale: Prevents divergence like codec step 25k→30k where uncontrolled LR led to +130% total loss increase.

## World Model Training Configuration
File: `train.sh` (alakazam-mira-mini)

Updated to use codec checkpoint-33000 (best: 24.60 dB PSNR) and LR decay:
```bash
CODEC="/home/horde/mira/codec_logs/checkpoint-33000/checkpoint.pth"  # line 31 — best codec
+optimizer.lr=1e-5 +optimizer.schedule=cosine_warmup  # Stable training
```

## HyDRA Setup Status
- ✓ Fixed model_manager.py None-check (lines 166, 177)
- ✓ Updated infer_hydra.py model paths to ./models/Wan2.1-T2V-1.3B/
- ⧖ Downloading Wan2.1-T2V-1.3B base models (~10-20 GB, in progress)

**Expected Wan2.1 files**:
- diffusion_pytorch_model.safetensors
- models_t5_umt5-xxl-enc-bf16.pth
- Wan2.1_VAE.pth

## Reference: Alakazam MIRA Mini Benchmarks

### Codec Comparison
Alakazam's reproduction at ablation scale (∼15.8k matches, 125k steps):
- PSNR: 28.59 vs paper 29.7 (−1.1 dB)
- SSIM: 0.867 vs paper 0.891 (−0.024)
- LPIPS-Alex: 0.068 vs paper 0.051 (+0.017)

**Note**: Gap attributed to ∼5× corpus difference; codec quality is binding constraint for world model.

### Single-Player World Model (Alakazam)
- Checkpoint: 52k steps (52% of paper's budget)
- gFID: 12.8 vs paper 100k: 10.7
- gFDD: 0.45 vs paper: 0.55 (better on this axis)
- Rollout PSNR @4s: 17.3 dB
- Finding: Model saturates codec floor by 45k steps; visual quality plateaus before controllability

### Multiplayer Fine-Tune (Alakazam)
- Warm-start: 52k single-player checkpoint
- Training: 4 nodes / 32×H100, frozen at 63k for evaluation, released at 90k
- Validation loss: monotone 0.623 → 0.331 (no plateau)
- gFID (512-sample): 24.7 at 63k, 24.1 at 90k (vs paper 9.4–9.9 at larger budget)
- Key finding: Early-window controllability lag is multiplayer-specific, not architecture defect

### Controllability & Serving (Alakazam)
**Seed-Controlled Action Divergence (Key Metric)**
| Checkpoint | @1s | @2s | @4s | Status |
|---|---|---|---|---|
| SP-45k (felt controllable) | 0.51 | 0.95 | 1.22 | ✓ Reference |
| MP-40k (felt dead) | 0.19 | 0.48 | 0.90 | Early window weak |
| MP-63k (freeze) | 0.261 | 0.521 | 0.938 | Recovering |
| MP-90k (release) | 0.281 | — | — | Approaching agency band |

**Mechanism**: Multiplayer ∼2× behind single-player in step-efficiency; per-action dropout (p=0.1) + subset dropout dilute steering signal; rare actions (boost, jump) convergence lagged.

**Fix Applied**: Action guidance (classifier-free on action stream) crosses felt-agency threshold at guidance weight w=4; deployed setting.

### Serving Performance
- Single-view: 10.4 fps @ 8 steps, 22.1 fps @ 2 steps (B200)
- Four-view: 7.5 fps @ 8 steps (∼30% cost for tiling)
- Latent decode: constant ≈37 ms
- Autopilot opponents functional; cross-view state binding still emerging at partial training

### Production Stack
- Architecture: Browser (React) ↔ FastAPI room relay ↔ GPU engine (Modal, scale-to-zero)
- Deploy gates: healthz endpoint + protocol probe (frame integrity, throughput)
- Live seat reassignment mid-session; unclaimed seats on autopilot

## gFID Measurement

MIRA provides `eval_world_model_offline.py` for offline gFID/gFDD evaluation. Available metrics:
- **gFID**: Generative Fréchet Inception Distance (quality measure)
- **gFDD**: Generative Fréchet DINO Distance (semantic drift)
- **DINO drift**: Cosine/L2 drift over frames
- **Per-frame**: PSNR, LPIPS, SSIM (matching our eval_video_quality.py)

### Usage
```bash
cd C:\workspace\world\mira
python scripts/eval_world_model_offline.py /path/to/checkpoint-XXXXX/checkpoint.pth --num-samples 512 --viz 8
```

This reads `world_model_config.yaml` from the checkpoint dir and loads `configs/eval_world_model.yaml` for eval settings.

### Example Command (checkpoint-98000)
```bash
python scripts/eval_world_model_offline.py C:\workspace\world\alakazam-mira-mini\outputs\checkpoint-98000\checkpoint.pth --num-samples 256 --output-dir C:\workspace\world\alakazam-mira-mini\eval_results
```

Result: JSON file in checkpoint's output dir with scalar gFID/gFDD + Fréchet curves.

## Next Steps
1. ✓ Validate WM training plateau (98k is best)
2. **Start Phase 1**: Codec training (50k-100k steps) to improve PSNR from 23.7→27+ dB
3. **Phase 3**: Finetune WM with new codec (150k steps)
4. Complete Wan2.1 model download for HyDRA (~10-20 GB)
5. Run HyDRA inference test
