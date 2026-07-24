# Training MIRA Mini on RacerX (local notes)

Local workflow for training the MIRA world model on RacerX playtest data — both
**warm-starting** (finetune) from the published `alakazam-mira-mini` weights and
training **from scratch** (codec and/or world model). This is **not** part of the PyPI
package (see [README.md](README.md) for playing the released model) — it documents the
local train/eval loop.

- **Finetune** (recommended single-GPU path) → "Launch a finetune" below.
- **From scratch** (codec, WM, or both) → "Training from scratch" below.

## Placeholders

Paths below use these placeholders — substitute your own checkout / cache locations
(the bats already hardcode them, so you rarely type these by hand):

| placeholder | what it is |
|---|---|
| `<mini>` | this repo (`alakazam-mira-mini`) — weights source + runner bats |
| `<mira>` | the mira trainer repo (`train_world_model.py`, `finetune.bat`, `smoke_test.bat`) |
| `<rx>` | the RacerX data pipeline (`racer-x/tools/replay-processing`, bats under `scripts\`) |
| `<rec>` | the recordings/output root (holds `mira_wds\`, `playtests\`, `mira\`) |
| `<hf>` | the Hugging Face cache (`~/.cache/huggingface/hub`) |
| `<snap>` | the downloaded snapshot dir under `<hf>/models--alakazamworld--mira-mini/snapshots/` |

Run each bat from its own repo (paths inside are relative to that repo).

## Runner bats (`<mini>`)

| bat | what it does |
|---|---|
| `run_mira.bat` | launch the released model to *play* in the browser |
| `download_weights.bat [1b\|364m\|all]` | pre-download weights into the HF cache (default `1b`) |
| `tensorboard.bat [logdir] [port]` | serve TensorBoard for training runs (default `<rec>\mira_wds` on :6006) |

## Which model to download for finetuning

**`alakazamworld/mira-mini`** (the `1b` — right for a discrete GPU / RTX 5090):

```
download_weights.bat 1b
```

That one repo bundles **both** checkpoints the world-model trainer needs, under
`<snap>` (i.e. `<hf>/models--alakazamworld--mira-mini/snapshots/<snap>/`):

- **World model** (finetune *from* this): `checkpoint-52000/checkpoint.pth`
- **Codec** (RAEv2, required to operate on latents; stays frozen): `codec/checkpoint-125000/checkpoint.pth`

`364m` = `alakazamworld/mira-mini-364m`, the laptop/Apple-silicon tier — not used here.

## Launch a finetune

From `<mira>`, `smoke_test.bat` already wires the codec checkpoint and the RacerX
index, and warm-starts from `checkpoint-52000` via `finetune_from` (see `finetune.bat`,
which hardcodes the `<snap>` path so you don't type it):

```
smoke_test.bat                                          200-step smoke, validate first
smoke_test.bat run.steps=50                             shorter
smoke_test.bat validation.val_n_samples=8 world_model_metrics.num_samples=16   fast (see below)
```

To warm-start explicitly through `<rx>\scripts\train_wm_smoke.bat` instead, point
`run.continue_from` at the checkpoint under `<snap>`:

```
train_wm_smoke.bat 50 run.continue_from=<snap>/checkpoint-52000/checkpoint.pth
```

- `finetune_from` = load **model weights only** (fresh optimizer/step counter) — a warm start.
- `continue_from` = resume **optimizer + step** too.

**Always check `nvidia-smi` first** — training crashes under GPU contention.

## Training from scratch

Two independent "from scratch" targets — the **codec** and the **world model** —
because the WM always trains on a **frozen** codec. You need a codec *first*; then you
either finetune the WM (above) or train it from scratch on that codec.

### 1. World model from scratch (random-init DiT, frozen mira-mini codec)

`finetune.bat` = `train.bat` + `run.finetune_from`. So **`train.bat` on its own IS the
from-scratch WM launcher** — `finetune_from`/`continue_from` both default to `null`, so
the 1.3 B DiT starts random-initialized. It still loads the released mira-mini codec
(`checkpoint-125000`, frozen) to turn frames into latents. From `<mira>`:

```
get_data.bat                       once: writes data_paths.bat (TRAIN_INDEX/TEST_INDEX)
train.bat run.steps=200            from-scratch WM smoke on the frozen codec
train.bat run.steps=20000 dataloader.num_workers=4 run.batch_size=2   a real run
```

Scaling knobs (train.bat header):
- Full 1B config: append `model/latent_world_model=1b`
- 4-player: append `model=multi_wrapper_world_model dataset.n_players=4`

Point it at RacerX instead of the default rocket-science index by appending
`dataset.train_index=<rec>/mira_wds/train/index.json dataset.test_index=...`
(or use `smoke_test.bat`, which already defaults to the RacerX index).

> From-scratch WM needs **far more data/steps** than a finetune to reach the same
> quality — warm-starting from `checkpoint-52000` is the sane single-GPU path.
> Train from scratch only to reproduce the recipe or when the architecture diverges
> from the released checkpoint.

### 2. Codec from scratch (RAEv2: DINOv3 encoder + learned decoder)

The codec is trained by `<mira>\scripts\train_codec.py`. The from-scratch entry is
`train_mira_smoke.bat` in `<rx>\scripts\` (no external checkpoint needed). Prereqs: the
**Meta-gated DINOv3-L/16** backbone at
`<mira>\dino_weights\dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth` (or set
`RS_DINO_WEIGHTS_DIR`), and a free GPU.

```
train_mira_smoke.bat 50
train_mira_smoke.bat 20000 run.batch_size=2
```

Output codec checkpoint lands at `<rec>/mira_wds/codec_smoke/checkpoint-*/checkpoint.pth`.
Feed it to the WM trainer to train the world model on **your own** codec instead of
mira-mini's:

```
train.bat model.architecture.config.codec_checkpoint=<rec>/mira_wds/codec_smoke/checkpoint-XXXX/checkpoint.pth run.steps=20000
```

### Full from-scratch pipeline (both models)

```
1. train_mira_smoke.bat <steps>          # codec from scratch  -> codec_smoke/checkpoint-*/checkpoint.pth
2. train.bat model.architecture.config.codec_checkpoint=<that path> run.steps=<steps>   # WM from scratch on it
```

Only do this to reproduce MIRA end-to-end; for RacerX experiments, the released codec +
a WM finetune is cheaper and better on limited data.

### Validation is the upfront cost

`smoke_test.bat` sets `validation.val_first=true` but does **not** cap
`validation.val_n_samples` (default **1024**) or `world_model_metrics.num_samples`
(default **2048**, each a 20-frame autoregressive rollout). So it validates *before*
step 1, which stalls for ~1–3 GPU-hours before any TensorBoard scalar appears — it is
**not** hung. For a fast smoke that logs within a minute or two, cap both:
`validation.val_n_samples=8 world_model_metrics.num_samples=16`.

## TensorBoard

Logging is ON by default (`tensorboard.logdir=${run.output_dir}/tb`). `smoke_test.bat`
writes to `<mira>\train_world_model_logs\`; the racer-x smoke bats write under
`<rec>\mira_wds\{codec_smoke,wm_smoke}\`.

```
tensorboard.bat                                    serve <rec>\mira_wds :6006
tensorboard.bat <mira>\train_world_model_logs      the smoke_test run
```

Uses mira's `.venv\Scripts\tensorboard.exe` (`python -m tensorboard` has no `__main__`).

## Training details (what actually trains)

Two-part latent world model:

- **Codec (frozen, ~900 M):** RAEv2 — a **DINOv3-L/16** encoder + learned decoder.
  Only encodes frames → latents (and decodes for viz). Loaded from `codec/checkpoint-125000`.
- **World model (trained, ~1.2 B):** `LatentWorldModel`, a **causal diffusion transformer**
  operating on codec latents. `hidden_dim 2048`, `16 layers`, GQA (`16` query / `4` KV heads),
  temporal attention every 4th block, `causal=true`, ~39 context frames, 288×512 @ 20 fps.
  Adaptive-LN action/timestep conditioning; `dropout_action_prob=0.1` for classifier-free guidance.

**Objective:** given ~39 past frames' latents + the pressed key, denoise the next frame's
latent (diffusion). **Actions (9-key vocab, load-bearing order):**
`W A S D Q E Space LShiftKey LControlKey` (RacerX `r`/flip → `Space`).

**Optim:** AdamW `lr 1e-4`, `betas (0.9, 0.99)`, `wd 0.1`, `warmup 1000`, EMA `0.9999`,
batch 1, `compile=false`.

**Eval metrics:** autoregressive rollout — seed 38 frames, generate 20, measure **drift**
(divergence over 20 frames) and **FDD** (Fréchet distance of generated vs real latents),
plus 8 decoded viz clips.

## Data

RacerX playtest → mira WebDataset. Current state:

- All 663 clips have npz + physics + mira samples; **video mp4 encodes are the gate**
  (disk-bounded `stream_video`). Only clips with a `video_720p.mp4` become trainable latents.
- Full capture ≈ **42 h** of gameplay @ 20 fps (663 clips); trainable subset grows as encodes land.
- Build/refresh + validate the WebDataset: `<rx>\scripts\rebuild_dataset.bat`
- Index consumed by the trainer: `<rec>\mira_wds\train\index.json`

> Finetuning on a *few* clips overfits fast — wait for more of the pending mp4
> encodes before a real (non-smoke) run.
