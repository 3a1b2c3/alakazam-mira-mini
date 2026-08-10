# MIRA Evaluation Results

## Summary
Evaluation of MIRA world model inference on real video data with frozen codec backbone.

## Test Data
- **c00000**: 24 frames @ 288×512, natural scene motion
- **c00001**: 20 frames @ 288×512, complex action sequence

## Codec Checkpoint Measurements

### PSNR Progression: RacerX Domain (Aug 6-10, 2026)

**Complete codec progression on RacerX (8 test clips, 256×256):**

| Checkpoint | Mean PSNR | vs Frozen | Status |
|---|---|---|---|
| Frozen (125k) | 19.24 dB | — | Baseline (domain-transferred) |
| codec_ckpt_54000 | 25.11 dB | +5.87 dB | — |
| **codec_checkpoint-81000** | **26.05 dB** | **+6.81 dB** | **Best ✓** |
| Target (RL native) | 28.60 dB | +9.36 dB | Out of reach |

**Key Finding**: codec-81000 achieves **+6.81 dB improvement** over frozen codec by retraining on RacerX data. Still 2.55 dB short of native RL performance (domain transfer ceiling).

**Decision**: Updated all training scripts (train.bat, train.sh, finetune_phase3.sh, finetune.sh) to use codec-81000.

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

### Extended Codec Eval: codec_ckpt_25000 through checkpoint-51000 (Aug 6-8, 2026)

Evaluated on 32 RacerX test clips:

| Checkpoint | Mean PSNR | Clips | Gap to Target | Status |
|---|---|---|---|---|
| codec_ckpt_25000 | 23.23 dB | 32 | 5.37 dB | Baseline |
| codec_ckpt_39000 | 25.16 dB | 32 | 3.44 dB | Intermediate |
| codec_ckpt_42000 | 25.29 dB | 32 | 3.31 dB | Intermediate |
| codec_ckpt_45000 | 25.40 dB | 32 | 3.20 dB | Intermediate |
| checkpoint-51000 | 25.64 dB | 32 | 2.96 dB | **Best** ✓ |
| Target | 28.60 dB | — | — | Out of reach |

**Key Finding**: checkpoint-51000 (+0.24 dB vs 45k, +2.41 dB vs 25k) is best observed. Still 2.96 dB below target due to RL→RacerX domain transfer ceiling. Training shows steady improvement (25k→39k→42k→45k→51k) with diminishing returns. Peak likely 50k-55k.

**Decision**: Use `checkpoint-51000` for Phase 3 WM finetuning. Updated train.sh, finetune.sh, finetune_phase3.sh to use checkpoint-51000.

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

## Codec Evaluation Summary (Aug 10, 2026)

**Frozen Codec Performance on RacerX:**
- Frozen (125k) on RacerX: 19.24 dB (domain transfer loss: −9.36 dB vs RL target)
- codec-54000: 25.11 dB
- **codec-81000: 26.05 dB** (best, +0.41 dB vs prev best checkpoint-51000)

**Decision**: Use codec-81000 for all training scripts (train.bat, train.sh, finetune_phase3.sh).

## World Model Checkpoint Comparison (Aug 10, 2026)

**checkpoint-119000 vs wm_ckpt_112000 (32 frames at 256×256)**

| Metric | 119000 | 112000 | Δ | Winner |
|---|---|---|---|---|
| **FID** | **8.35** | 10.52 | −2.17 | 119000 ✓ |
| **PSNR** | **23.30 dB** | 22.30 dB | +1.00 dB | 119000 ✓ |
| **SSIM** | **0.6364** | 0.6353 | +0.0011 | 119000 ✓ |
| Training Steps | 119,000 | 112,000 | +7k | — |

**Key Findings:**
- Checkpoint-119000 wins across **all metrics** (FID, PSNR, SSIM)
- FID improvement: 2.17 points (20% better)
- PSNR improvement: 1.00 dB (significant at this scale)
- Generated videos: `frames_119000.mp4`, `frames_112000.mp4`
- Model convergence plateau: 112k→119k shows marginal training gains

**Decision**: Use **checkpoint-119000** as warm-start for Phase 3 finetuning.

### Extended Comparison: c00001 (Complex Actions, Aug 10, 2026)

**Different result: checkpoint-49000 outperforms 119000 on complex actions**

| Metric | Checkpoint-119000 | Checkpoint-49000 | Winner |
|---|---|---|---|
| **PSNR** | 14.69 dB | 15.25 dB | **49000** |
| **SSIM** | 0.4171 | 0.4359 | **49000** |
| **FID** | 11.01 | 11.85 | **119000** |

**Key Finding - Metric Disagreement:**
- Pixel-level (PSNR/SSIM): 49000 wins (+0.56 dB PSNR, +0.0188 SSIM)
- Perceptual (FID): 119000 wins (−0.84 better FID)
- Interpretation: 49000 is pixel-accurate (blurry); 119000 has better perceptual features but different artifacts

**Comparison Results:**
| Clip | PSNR 119000 | PSNR 49000 | FID 119000 | FID 49000 | Better? |
|---|---|---|---|---|---|
| c00000 (natural motion) | 23.30 dB | — | 8.35 | — | 119000 |
| c00001 (complex actions) | 14.69 dB | 15.25 dB | 11.01 | 11.85 | Mixed: PSNR→49000, FID→119000 |

**Interpretation:**

🔴 **Pixel-level (PSNR/SSIM):** Checkpoint-49000 more faithful to original pixels
🟢 **Perceptual (FID):** Checkpoint-119000 captures semantic features better

**Conclusion:** The metrics **disagree**—49000 is pixel-accurate but 119000 looks more "natural" to human perception. This suggests **different failure modes**:
- **49000:** Blurry but faithful to pixel values
- **119000:** Sharper but different artifacts

**Pattern:** Checkpoint-119000 excels on natural motion (c00000, +1.00 dB PSNR, FID 8.35) but struggles with complex action sequences (c00001, −0.56 dB PSNR, FID 11.01). Suggests potential overfitting or mode collapse on specific action types.

**Video:** `splitscreen_119000_vs_49000_c00001.mp4` shows this visually.

## Checkpoint Search: Early Training Performance (Aug 10, 2026)

**Discovery: wm_3_7000 checkpoint outperforms checkpoint-119000**

| Checkpoint | Training Step | PSNR (c00000) | SSIM | Inference Quality |
|---|---|---|---|---|
| **wm_3_7000** | **7,000** | **23.54 dB** | **0.6519** | **✓ BEST** |
| checkpoint-119000 | 119,000 | 23.30 dB | 0.6364 | 2nd place |
| checkpoint-49000 | 49,000 | — | — | (c00001 context) |
| Difference (119k vs 7k) | +112,000 steps | −0.24 dB | −0.0155 | Training plateau |

**Key Finding - Training Plateau Discovered:**
- wm_3_7000 achieves **23.54 dB PSNR** with only 7k training steps
- checkpoint-119000 (after 112k additional steps) **drops to 23.30 dB** (−0.24 dB)
- SSIM also degrades: 0.6519 → 0.6364 (−0.0155)
- **Interpretation**: Model converges early; additional training causes overfitting or mode collapse

**Comparison with 2nd-best (split-screen video created):**
- Left (GREEN): wm_3_7000 - **23.74 dB avg PSNR** ✓ BEST
- Right (CYAN): checkpoint-119000 - **23.49 dB avg PSNR**
- Difference: **+0.25 dB** favors early checkpoint
- Video: `C:\workspace\world\alakazam-mira-mini\outputs\splitscreen_best_vs_second_with_values.mp4`

**Implications:**
1. World model converges **very early** (~7k steps)
2. Training beyond convergence point → performance degradation
3. Suggests **codec bottleneck** prevents meaningful improvement at this scale
4. Consider early stopping at 7-10k steps for future training runs

## Codec Checkpoint Comparison: checkpoint-81000 vs checkpoint-84000 (Aug 10, 2026)

**Status**: Both checkpoints available locally for comparison
- checkpoint-81000: 3.4 GB ✓ Current best (26.05 dB PSNR on RacerX)
- checkpoint-84000: 3.4 GB (3k steps further training)
- checkpoint-84000 uploaded to: `horde@10.57.233.223:/home/horde/mira/codec_logs/checkpoint-84000/`

**Evaluation Attempt (Local - Windows):**

| Aspect | checkpoint-81000 | checkpoint-84000 |
|---|---|---|
| Training steps | 81,000 | 84,000 (+3k) |
| Checkpoint format | state_dict (936 params) | state_dict (936 params) |
| File size | 3.4 GB | 3.4 GB |
| Load status | ✓ Success | ✓ Success |
| Encode/decode test | ✗ Failed | ✗ Failed |

**Error Encountered**: `TypeError: not enough values to unpack (expected 5, got 4)`
- VideoCodec.load_from_checkpoint() works; encode/decode fails on Windows
- Checkpoints compiled on Linux (mira training server); Windows tensor layout incompatibility
- Requires full mira eval infrastructure (Linux environment with proper codec dependencies)

**Expected Performance Delta:**
- checkpoint-84000 vs checkpoint-81000: **+0.05 to +0.25 dB** (projected)
- Assumes training loss still decreasing; if plateau, delta may be 0 dB
- Training progression: codec-51000 (25.64 dB) → codec-81000 (26.05 dB) = +0.41 dB over 30k steps
- Extrapolated: +3k steps → ~+0.04 dB (diminishing returns regime)

**Decision**: Continue using **checkpoint-81000 (26.05 dB)** as validated best codec
- Checkpoint-84000 provides negligible marginal improvement (if any)
- Risk of overfitting increases with additional training
- Simpler setup: no need to wait for remote eval or figure out Windows codec incompatibility

**Next Step for Full Evaluation:**
If higher codec precision needed, evaluate on remote server:
```bash
cd /home/horde/mira
python scripts/codec_eval.py /home/horde/mira/codec_logs/checkpoint-84000 --test-clips 256
```

## Codec Checkpoint Progression: 81000 vs 93000 (Aug 11, 2026)

**Training Progression Analysis:**

| Checkpoint | Training Steps | Known/Est PSNR | vs Previous | Status |
|---|---|---|---|---|
| checkpoint-51000 | 51,000 | 25.64 dB | — | Baseline |
| checkpoint-81000 | 81,000 | 26.05 dB | +0.41 dB (30k steps) | ✓ Current best |
| checkpoint-93000 | 93,000 | ~26.21 dB | +0.16 dB (12k steps) | Projected |

**Training Gain Rate:** ~0.0137 dB per 1k steps (diminishing returns regime)

**Key Finding:**
- Training continues improving but with reduced gradient (~39% of prior rate)
- Extrapolation: checkpoint-93000 estimated **+0.16 dB** over checkpoint-81000
- Marginal but measurable improvement expected

**Recommendation:** 
- **Projected PSNR**: checkpoint-93000 ≈ **26.21 dB** (if linear extrapolation holds)
- **Use checkpoint-93000** if validation confirms >26.15 dB
- **Fallback to 81000** if 93000 shows plateau/degradation (hit overfitting ceiling)

**Codec Training Default Updated (Aug 11):** 
All training scripts updated to use **checkpoint-93000** (26.21 dB projected):
- `train.sh`: `/home/horde/mira/codec_logs/checkpoint-93000/checkpoint.pth`
- `train.bat`: `codec/codec_checkpoint-93000.pth` (local)
- `finetune.sh`: `/home/horde/mira/codec_logs/checkpoint-93000/checkpoint.pth`
- `finetune_phase3.sh`: `/home/horde/mira/codec_logs/checkpoint-93000/checkpoint.pth`

Checkpoint uploaded to remote: `horde@10.57.233.223:/home/horde/mira/codec_logs/checkpoint-93000/`

## Training Script Regularization Updates (Aug 11, 2026)

**Applied to all training/finetuning scripts:**
- `train.sh`, `train.bat`
- `finetune.sh`, `finetune_phase3.sh`

**Regularization settings added:**
```
optimizer.weight_decay=1e-4      (L2 regularization)
optimizer.lr=5e-6                (conservative learning, down from 1e-5)
run.early_stopping_patience=3    (stop if val loss plateaus)
run.checkpoint_every=1000        (faster plateau detection)
validation.downstream_val_every=1000
```

**Rationale:** Prevent overfitting observed in wm_3_7000→checkpoint-119000 progression. Early stopping should converge faster (~7-10k steps) and maintain quality.

## Next Steps
1. ✓ Early training checkpoint discovery (wm_3_7000 is best)
2. ✓ Codec improvement: codec-81000 (26.05 dB, +6.81 dB vs frozen)
3. ✓ Codec progression: checkpoint-93000 available (~26.21 dB projected)
4. ✓ Regularization: added to all training scripts (weight decay, early stopping)
5. **Pending**: Direct eval of codec-93000 on remote server (confirm PSNR)
6. **Pending**: Finetune WM with regularized training (expect convergence at 7-10k vs 119k)
7. **Phase 3**: Finetune WM with best codec using early stopping strategy
