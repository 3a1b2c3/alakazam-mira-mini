# alakazam-mira-mini

Play **MIRA Mini**, a neural world model of car soccer, locally on your own GPU.
Every frame is generated live by the model. After the first weight download,
everything runs on your machine; there is no cloud dependency and no account.

```
pip install alakazam-mira-mini
mira-mini play
```

Useful flags: `--checkpoint PATH` (play your own finetune), `--interp` (2x display interpolation), `--steps N`, `--no-fast`, `--verbose`.

MIRA Mini is our from-scratch reproduction of the MIRA recipe
([General Intuition](https://www.generalintuition.com/) × [Kyutai](https://kyutai.org/),
with Epic Games), compressed until it runs on consumer hardware: fewer diffusion steps,
a smaller student model, and a compact decoder. Measurements and method are in the
[technical report](https://alakazam.gg/mira-mini).

## What you need

- An NVIDIA GPU (CUDA), or a Mac with Apple silicon (M1 or newer). CPU-only machines
  are not supported; generation is too slow to play.
- Disk for the weights, downloaded once from Hugging Face: ~5 GB for the 364M model,
  ~12 GB for the 1B.
- The weight repositories are public on Hugging Face
  ([alakazamworld](https://huggingface.co/alakazamworld)); the first run downloads them
  automatically.

## Picking a model

`mira-mini play` chooses weights for your machine: **CUDA gets the 1B**, **Apple silicon
gets the 364M laptop tier** (an MLX transformer + Core ML decoder, ~8 fps on a 2021 M1 Pro).
Override it:

```
mira-mini play --model 1b     # the 1B single-player model (needs a discrete GPU)
mira-mini play --model 364m   # the laptop tier, 
```

Play your **own finetune** instead of the released weights — the `--model` bundle
is still fetched for its codec + context, only the DiT weights are swapped:

```
mira-mini play --model 1b --checkpoint C:\path\to\outputs\finetune_ch\checkpoint-49000\checkpoint.pth
```

The checkpoint must be the **same architecture** as `--model` (i.e. a finetune of it).

## Options

| flag / env | effect |
|---|---|
| `--model {auto,1b,364m}` | which weights to run (default: auto, by device) |
| `--checkpoint PATH` | play a custom `checkpoint.pth` (your finetune) with the `--model` codec/context |
| `--steps N` | sampler steps; 2 is the steadier default, 1 is smoother but drifts more |
| `--port N` | web UI port (default 8770) |
| `--no-browser` | don't open the browser automatically |
| `MIRA_HF_REPO` | use a custom Hugging Face weights repo |
| `MIRA_HOME` | where bundles are cached (default `~/.cache/alakazam-mira`) |
| `MIRA_DEVICE` | force `cuda` / `mps` / `cpu` |

## Weights and license

Model weights live on Hugging Face under
[alakazamworld](https://huggingface.co/alakazamworld) and are **CC BY-NC-SA 4.0**,
inherited from the training dataset
([kyutai/rocket-science](https://huggingface.co/datasets/kyutai/rocket-science), Rocket
League content used with Epic Games' permission). Non-commercial, share-alike, with
attribution. The model is a research demonstration; long rollouts drift from exact
physics.

## Credits

The architecture, training recipe, and dataset are General Intuition's and Kyutai's,
released openly with Epic Games ([mira-wm/mira](https://github.com/mira-wm/mira)).
MIRA Mini is Alakazam's independent reproduction and compression of that work. The
weights are an independent release by Alakazam: not released by, associated with, or
endorsed by General Intuition, Kyutai, or Epic Games.


## 0.1.3

Packaging fix: 0.1.1 and 0.1.2 wheels were missing the engine (`mira_vm`), the room relay, and the vendored `mira` inference runtime, so `mira-mini play` crashed with ModuleNotFoundError after downloading weights. 0.1.3 ships all of them. Thanks to the first player who reported it.

## 0.1.4

Local-play polish from a live rehearsal: the access-key prompt no longer appears (the local relay never checked it; the page now pre-seeds it), and few-step bundles (364m, psd) default to 2 diffusion steps via the engine's hard override, so `mira-mini play` hits its rated frame rate without flags.

## 0.1.5

The access-key bypass now targets the store the UI actually reads (sessionStorage) and the play URL carries `key=local`, so the prompt is gone on every browser and cache state.

## 0.1.6

The Apple fast stack is now automatic: on a Mac with the 364m bundle, `mira-mini play` wires the MLX transformer (whole-step compiled), the Core ML decoder (pipelined in a child process), and 2x display interpolation, roughly doubling the delivered frame rate; `--no-fast` restores plain torch. The CLI shows staged loading and opens the browser only when the model is ready, and the room switches to the play screen only after the first generated frame arrives. Best served by python 3.12 (coremltools has no 3.13+ bindings yet; without them the fast stack silently falls back to torch).

## 0.1.7

The play URL is a clickable terminal hyperlink (OSC 8).

## 0.1.8

The room start handshake retries with backoff instead of failing once when the engine slot is briefly held by a previous session or an idle-closed socket.

## 0.1.9

The banner shows the installed version.

## 0.1.10

Animated loading spinner with elapsed time; engine logs are hidden by default (`--verbose` restores them).

---

# Training & Finetuning Guide

## Current Status

**Codec Training (Step 1):** ✅ Active
- Checkpoint-25000: **Latest** (TBD dB PSNR)
- Progress: 20.46 → 21.76 → 22.49 → 22.xx dB (continuing)
- Training continues on Horde (checkpoint every 5000 steps)
- scripts now use checkpoint-25000 (not frozen)

**World Model Finetuning (Step 2):** ⏭️ Ready to deploy
- Warm-start: mira-mini checkpoint-52000 (1B diffusion model)
- Codec: checkpoint-25000 (latest trained, not frozen)
- Data: RacerX mira_wds
- Command: `bash finetune.sh` on Horde

**Evaluation:** ✅ Comprehensive
- Primary: `codec_recon_psnr.py` (batch PSNR on test data)
- Encoder analysis: `eval_encoder_only.py` (192x compression bottleneck)
- Single-file: `eval_frozen_on_real_video.py` (Rocket League comparison)
- Shard eval: `eval_on_shard.py` (batch WebDataset evaluation)

---

## Key Finding: Encoder Bottleneck

**Frozen encoder compresses by ~60x**, limiting photorealism to ~13 dB PSNR.
- Encoder determines how much info is preserved (bottleneck)
- Decoder can only reconstruct what's in latent
- Training improves: frozen 12.84 dB → trained 16.26 dB (+3.4 dB)

**To improve photorealism:**
1. Larger latent (less compression)
2. Better encoder (more efficient compression)
3. More training (encoder learns better representations)

**Evaluation:** See `eval_encoder_only.py` for split-screen test set analysis.

---

## Codec Training (Step 1)

Train a RAEv2 video codec on RacerX data.

```bash
# On Horde (from alakazam-mira-mini/)
bash train_codec.sh

# Custom steps
bash train_codec.sh run.steps=200000 run.batch_size=2
```

**Progress:**
- Checkpoint-7000: 20.46 dB PSNR
- Checkpoint-10000: 21.76 dB PSNR (+1.3 dB)
- Checkpoint-15000: 22.49 dB PSNR (+0.73 dB)
- Checkpoint-25000: TBD dB PSNR (in progress) ✅
- Target: ~28.6 dB (continuing...)

**Evaluate locally:**
```powershell
python codec_recon_psnr.py \
  --data C:\recordings\mira_wds\train\index.json \
  --codec outputs\codec_ckpt_7000.pth \
  --num-samples 32
```

## World Model Finetuning (Step 2)

Finetune 1B diffusion model (mira-mini checkpoint-52000) on RacerX.

```bash
# On Horde
scp finetune.sh horde@10.57.233.223:/home/horde/alakazam-mira-mini/
ssh horde@10.57.233.223 "cd alakazam-mira-mini && bash finetune.sh"
```

**Uses:**
- Warm-start: mira-mini checkpoint-52000
- Codec: checkpoint-25000 (trained, not frozen)
- Data: RacerX WebDataset (mira_wds)

**Local exploration:**
```powershell
# Inspect checkpoint
.\.venv\Scripts\python.exe inspect_checkpoint.py

# Visualize input/target frames
.\.venv\Scripts\python.exe infer_world_model.py

# Interactive player
.\.venv\Scripts\python.exe interactive_wm.py
# Commands: load 0, show, export, list, info, quit
```

## Checkpoints

| Name | Type | Path | Size | Quality |
|------|------|------|------|---------|
| codec-7000 | Codec | `codec_logs/checkpoint-7000/` | 3.6 GB | 20.46 dB |
| codec-10000 | Codec | `codec_logs/checkpoint-10000/` | 3.6 GB | 21.76 dB |
| codec-15000 | Codec | `codec_logs/checkpoint-15000/` | 3.6 GB | 22.49 dB |
| codec-25000 | Codec | `codec_logs/checkpoint-25000/` | 3.6 GB | TBD dB ✅ |
| wm-21000 | World Model | `outputs/scratch_ch/checkpoint-56000/checkpoint-21000-horde.pth` | 7.79 GB | Loss: 0.061 |
| wm-49000 | World Model | Baseline from step 2 | - | - |

## Playing Custom Finetune

Run locally after training:

```bash
mira-mini play --checkpoint C:\path\to\outputs\finetune_ch\checkpoint-49000\checkpoint.pth
```

## Scripts

### Codec Training
- `train_codec.sh` - Train codec from scratch (step 1)

### Codec Evaluation
- `codec_recon_psnr.py` - Batch PSNR evaluation (primary tool)
- `eval_codec_checkpoints.py` - Compare multiple checkpoints
- `eval_encoder_only.py` - Encoder bottleneck analysis (frozen vs trained)
- `eval_frozen_on_real_video.py` - Single Rocket League video comparison
- `eval_on_shard.py` - WebDataset shard batch evaluation
- `visualize_codec_real.py` - Real data visualization + split-screens

### World Model Training
- `finetune.sh` - Finetune world model (step 2, recommended)
- `train.sh` - Train from scratch (alternative)

### World Model Evaluation & Exploration
- `inspect_checkpoint.py` - Inspect checkpoint structure (state_dict, size, etc.)
- `infer_world_model.py` - Extract input/target frames from checkpoint
- `interactive_wm.py` - Menu-driven checkpoint explorer

### Utilities
- `download_checkpoint.py` - Download from Horde via SFTP
- `upload_checkpoint.py` - Upload to Horde via SFTP
- `test_codec.py` - Quick codec load/encode/decode test
- `eval_and_infer.bat` - Batch wrapper for evaluation + inference
