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
| `<mira>` | the mira trainer repo — `.venv`, `scripts/train_world_model.py`, configs. **Sibling of `<mini>`.** |
| `<rx>` | the RacerX data pipeline (`racer-x/tools/replay-processing`, bats under `scripts\`) |
| `<rec>` | the recordings/output root (holds `mira_wds\`, `playtests\`, `mira\`) |
| `<hf>` | the Hugging Face cache (`~/.cache/huggingface/hub`) |
| `<snap>` | the downloaded snapshot dir under `<hf>/models--alakazamworld--mira-mini/snapshots/` |

**All `.bat` launchers live in `<mini>` (this repo).** They derive the sibling mira repo
relatively (`%~dp0..\mira`) — no absolute repo paths — so you run them from `<mini>` while
they operate on `<mira>`'s `.venv`/`scripts`/configs (which `setup.bat` creates there). The
`.sh` equivalents stay in `<mira>` for cluster use.

## Runner bats (`<mini>`)

| bat | what it does |
|---|---|
| `run_mira.bat` | launch the released model to *play* in the browser |
| `download_weights.bat [1b\|364m\|all]` | pre-download weights into the HF cache (default `1b`) |
| `tensorboard.bat [logdir] [port]` | serve TensorBoard for training runs (default `<rec>\mira_wds` on :6006) |
| `setup.bat` | create `<mira>\.venv` (uv, py3.11, cu128 torch) + editable mira install |
| `get_data.bat` | download rocket-science shards + write `<mira>\data_paths.bat` |
| `download_models.bat` | download the rocket-science dataset into the HF cache |
| `train.bat` | low-level WM launcher on the frozen codec (from scratch; rocket-science data by default) |
| `finetune.bat` | `train.bat` + `run.finetune_from` (warm-start from `checkpoint-52000`) |
| **`train_racerx.bat [steps]`** | **real from-scratch** RacerX run (train+test split, resolved codec) → `wm_racerx` |
| **`finetune_racerx.bat [steps]`** | **real finetune** RacerX run (warm-start `checkpoint-52000`) → `wm_racerx_ft` |
| `smoke_finetune.bat` | short **finetune** smoke on RacerX (200 steps, capped validate-first) |
| `smoke_scratch.bat` | short **from-scratch** smoke on RacerX (200 steps, capped) |
| `smoke_test_all.bat` | run + PASS/FAIL every training path (see "Smoke-test every path") |
| `train_racerx.sh` | Linux RacerX launcher (`RX_ROOT`+`CODEC`, `WM=` to finetune) |

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

From `<mini>` (all bats live here), `smoke_finetune.bat` already wires the codec checkpoint
and the RacerX index, and warm-starts from `checkpoint-52000` via `finetune_from` (see `finetune.bat`,
which hardcodes the `<snap>` path so you don't type it):

```
smoke_finetune.bat                                          200-step smoke, validate first
smoke_finetune.bat run.steps=50                             shorter
smoke_finetune.bat validation.val_n_samples=8 world_model_metrics.num_samples=16   fast (see below)
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
from-scratch WM launcher** — `fine

tune_from`/`continue_from` both default to `null`, so
the 1.3 B DiT starts random-initialized. It still loads the released mira-mini codec
(`checkpoint-125000`, frozen) to turn frames into latents. From `<mini>`:

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
(or use `smoke_finetune.bat`, which already defaults to the RacerX index).

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

## Smoke-test every path

`<mira>` ships a suite that runs all five runnable paths (data-loader test, codec from
scratch, WM finetune, WM from scratch, full codec→WM chain), each **capped** so it
reaches a checkpoint in minutes, and reports PASS/FAIL by verifying each run actually
**wrote a checkpoint** (exit 0 alone isn't enough — a run can 0-exit after a stalled
validation). It runs all steps even if one fails.

Windows:
```
smoke_test_all.bat            full suite (20 steps each)
smoke_test_all.bat 10         fewer steps
```

Linux / WSL / cluster (env-driven paths, uses `pixi run --frozen`):
```bash
DATA_INDEX=<rec>/mira_wds/train/index.json OUT=<rec>/mira_wds \
CODEC=<snap>/codec/checkpoint-125000/checkpoint.pth WM=<snap>/checkpoint-52000/checkpoint.pth \
RS_DINO_WEIGHTS_DIR=<dino> ./smoke_test_all.sh
```

Output ends with a summary like:
```
  1 data loader test ......... PASS
  2 codec from scratch ....... PASS
  3 world model finetune ..... PASS
  4 world model from scratch . PASS
  5 full chain (codec->WM) ... PASS
```
Steps skip cleanly (`SKIP`) if their prereq env var is unset (e.g. no `RS_DINO_WEIGHTS_DIR`
→ codec step skipped). Run `smoke_test_all.sh` via `sbatch` to also exercise the SLURM path.

## Linux wrapper scripts

The `.bat` launchers are Windows-only. `<mira>` also has `.sh` equivalents for Linux/WSL/
cluster use (env-driven, no hardcoded paths, `pixi run --frozen` by default):

| script | mirrors | required env |
|---|---|---|
| `train_codec.sh` (`<mira>`) | `train_mira_smoke.bat` (codec from scratch) | `DATA_INDEX`, `RS_DINO_WEIGHTS_DIR` |
| `train.sh` (`<mira>`) | `train.bat`/`finetune.bat` (world model) | `DATA_INDEX`, `CODEC`; `TEST_INDEX`/`WM` optional |
| `smoke_test_all.sh` (`<mira>`) | `smoke_test_all.bat` (full suite) | `DATA_INDEX`, `OUT` (+ `CODEC`/`WM`/`RS_DINO_WEIGHTS_DIR` per step) |
| **`train_racerx.sh`** (`<mini>`) | RacerX Linux one-shot (wires train+test split → `train.sh`) | `RX_ROOT`, `CODEC`; `WM` optional |

```bash
# codec from scratch
DATA_INDEX=<rec>/mira_wds/train/index.json RS_DINO_WEIGHTS_DIR=<dino> ./train_codec.sh run.steps=20000

# WM from scratch (omit WM) or finetune (set WM=)
DATA_INDEX=<rec>/mira_wds/train/index.json CODEC=<snap>/codec/checkpoint-125000/checkpoint.pth \
  WM=<snap>/checkpoint-52000/checkpoint.pth ./train.sh run.steps=5000
```

Override the env prefix with `RUN=""` (torch on PATH) or `RUN="source .venv/bin/activate &&"`.
Extra Hydra overrides pass straight through (`"$@"`).

## Train RacerX on Linux (WSL / cluster)

The data is built on Windows (`<rec>\mira_wds`), but real training runs best on Linux (proper
NCCL/multi-GPU, faster dataloaders). `train_racerx.sh` (`<mini>`) is the one command: it points
at the RacerX **train + held-out test** split, the frozen mira-mini codec, an optional warm-start,
then calls `<mira>/train.sh`. Requires `<mini>` and `<mira>` cloned **side by side**.

**1. Environment** (Linux/WSL box, once):
```bash
git clone <mira-repo-url> mira && cd mira
export PATH="$HOME/.pixi/bin:$PATH"      # if pixi isn't on PATH
pixi install --locked                     # solve + fetch the env (no network at run time)
```
(or a plain venv with `pip install -e ".[hf,hydra,viz]"` + CUDA torch, then `RUN="source .venv/bin/activate &&"`.)

**2. Data** — the WebDataset is **portable** (shard paths in `index.json` are relative):
- **WSL:** read it in place at `/mnt/c/recordings/mira_wds` (slow 9p FS — fine to start; **copy to the
  Linux native FS for a real run**: `cp -r /mnt/c/recordings/mira_wds ~/mira_wds`).
- **Remote cluster:** `rsync -a /mnt/c/recordings/mira_wds/ user@host:~/mira_wds/` (or scp).

**3. Checkpoints** — get the mira-mini weights on the Linux box:
```bash
pixi run huggingface-cli download alakazamworld/mira-mini    # -> ~/.cache/huggingface/... 
```
Note the `<snap>` dir; `CODEC` = `<snap>/codec/checkpoint-125000/checkpoint.pth`,
`WM` = `<snap>/checkpoint-52000/checkpoint.pth`. DINOv3-L/16 (`RS_DINO_WEIGHTS_DIR`) only for codec training.

**4. Launch** (from `<mini>`):
```bash
# from scratch on RacerX (train + honest held-out eval):
RX_ROOT=~/mira_wds CODEC=<snap>/codec/checkpoint-125000/checkpoint.pth \
  ./train_racerx.sh run.steps=20000 run.batch_size=2 dataloader.num_workers=8

# finetune from mira-mini instead: add WM=
RX_ROOT=~/mira_wds CODEC=<snap>/codec/checkpoint-125000/checkpoint.pth \
  WM=<snap>/checkpoint-52000/checkpoint.pth ./train_racerx.sh run.steps=10000
```
`train_racerx.sh` auto-uses `<RX_ROOT>/test/index.json` for eval when present (the holdout), so metrics
are honest. Extra Hydra overrides pass straight through.

**5. Multi-GPU** — `train_racerx.sh` is single-process; for multi-GPU/multi-node use `<mira>/train.sh`
under `torchrun`, or the `<mira>/train.sbatch` SLURM launcher (next section) with
`TRAIN=<rec>/mira_wds/train/index.json TEST=<rec>/mira_wds/test/index.json`.

## Run on SLURM

Single-GPU/Windows is the local path; to scale out, `<mira>\train.sbatch` is a minimal
launcher wrapping `torchrun` (single-node multi-GPU **and** multi-node via c10d
rendezvous). No Docker needed — env comes from the repo's `pixi`. Requires Linux + NCCL.

### End-to-end steps

1. **Code + env** (Linux login node, for NCCL):
   ```bash
   git clone <mira-repo-url> mira && cd mira
   export PATH="$HOME/.pixi/bin:$PATH"     # if pixi isn't already on PATH
   pixi install --locked                    # solve + download the env ONCE (no network at job time)
   ```
2. **Data + checkpoints** onto the cluster (scp/rsync from the workstation, or re-download):
   - WebDataset: `mira_wds/train/index.json` + shard tar  →  `<rec>/mira_wds/train`
   - Codec + WM checkpoints: `pixi run huggingface-cli download alakazamworld/mira-mini`
     (gives `<snap>/codec/checkpoint-125000` and `<snap>/checkpoint-52000`)
   - DINOv3-L/16 weights (codec step only)  →  set `RS_DINO_WEIGHTS_DIR=<dino>`
3. **Smoke-test all paths first** (submit the suite; confirm all 5 rows PASS before a real run):
   ```bash
   DATA_INDEX=<rec>/mira_wds/train/index.json OUT=<rec>/mira_wds \
   CODEC=<snap>/codec/checkpoint-125000/checkpoint.pth WM=<snap>/checkpoint-52000/checkpoint.pth \
   RS_DINO_WEIGHTS_DIR=<dino> sbatch --wrap="./smoke_test_all.sh"
   ```
4. **Real training** via `train.sbatch` (edit the four `#SBATCH` resource lines first) — see below.
5. **Scale + monitor:** multi-node is just `#SBATCH --nodes=N` (rendezvous is automatic);
   watch with `squeue`, `tail -f slurm-<jobid>.out`, and TensorBoard on the run's `output_dir/tb`.

```bash
# once on the login node (solve + download the env, no network at job time):
pixi install --locked

# world model (add run.finetune_from=<snap>/checkpoint-52000/checkpoint.pth to warm-start):
CODEC=<snap>/codec/checkpoint-125000/checkpoint.pth \
TRAIN=<rec>/mira_wds/train/index.json TEST=<rec>/mira_wds/test/index.json \
  sbatch train.sbatch run.steps=20000 run.batch_size=2

# codec from scratch:
ENTRY=scripts/train_codec.py RS_DINO_WEIGHTS_DIR=<dino> TRAIN=... TEST=... sbatch train.sbatch
```

- Edit the four `#SBATCH` lines (`--nodes`, `--gpus-per-node`, `--cpus-per-task`, `--time`);
  multi-node just needs `--nodes=N` — rendezvous is automatic.
- Extra Hydra overrides pass straight through (`$*`).
- `pixi` must be on the **compute nodes'** PATH (the sbatch adds `~/.pixi/bin`), and
  `.pixi/envs/` must sit on a **shared filesystem** all nodes can read. Pre-solve once with
  `pixi install --locked`; the job uses `pixi run --frozen` so it never re-solves.
- Override the env prefix with `RUN=""` (torchrun on PATH) or `RUN="source .venv/bin/activate &&"`.

### Validation is the upfront cost

`smoke_finetune.bat` sets `validation.val_first=true` but does **not** cap
`validation.val_n_samples` (default **1024**) or `world_model_metrics.num_samples`
(default **2048**, each a 20-frame autoregressive rollout). So it validates *before*
step 1, which stalls for ~1–3 GPU-hours before any TensorBoard scalar appears — it is
**not** hung. For a fast smoke that logs within a minute or two, cap both:
`validation.val_n_samples=8 world_model_metrics.num_samples=16`.

## TensorBoard

Logging is ON by default (`tensorboard.logdir=${run.output_dir}/tb`). `smoke_finetune.bat`
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

## Reference values (what "good" looks like)

Absolute targets from the **MIRA Mini technical report** (Alakazam's 1B reproduction of MIRA
on Rocket League). **Caveat:** these are *Rocket League* numbers — RacerX values will differ, so
treat them as **scale anchors + trajectory shapes**, not literal pass marks. What transfers is the
*shape* (monotone val-loss descent, metrics improving over steps) and the *gates*.

**Codec** (125k steps, eval on 2,048 held-out clips, full-frame LPIPS-AlexNet):

| metric | reproduction @125k | paper @125k | gate |
|---|---|---|---|
| PSNR ↑ | 28.59 | 29.7 | "within 2 dB" → pass |
| SSIM ↑ | 0.867 | 0.891 | — |
| LPIPS-Alex ↓ | 0.068 | 0.051 | — |

**Single-player world model** (1B, frozen codec, 80-frame clips):
- Train loss: **9.39 from scratch** → warm-start opens at **0.70**.
- Val loss (256 samples): **1.176 → 0.385** over 5k → 50k steps; stable ≈0.08 train-val gap = no overfit.
- Generation @52k: **gFID 12.8** (paper 10.7 @100k), **gFDD 0.45** (paper 0.55 — lower=better), rollout **PSNR@4s 17.3**, **LPIPS@4s 0.32**.
- FDD decomposition: WM-above-codec **0.21** < codec floor **0.26** → model is *codec-limited* (better codec is the next lever, not a bigger WM).

**Multiplayer fine-tune** — val loss (256 samples), monotone, ≈0.08 gap:

| step | 5k | 10k | 20k | 30k | 40k | 50k | 60k | 80k |
|---|---|---|---|---|---|---|---|---|
| val loss | 0.623 | 0.516 | 0.438 | 0.393 | 0.361 | 0.340 | 0.331 | 0.320 |

**Controllability** (action authority = seed-controlled action-divergence @1 s; ~0.30 = felt-agency band):
- SP ladder @1s: action-deaf ≤10k, inflects 10k→20k, climbs to **0.51** @45k.
- Paper's ARR (1B): **0.63 @25k → 0.85 @50k → 0.90 @100k** — controllability converges *after* visual quality.

**How to use these:** run validation at step 0 (`val_first=true`) to get your RacerX baseline, then
confirm loss descends monotonically and eval metrics move the right way (↓ drift/FDD/LPIPS,
↑ PSNR/SSIM). The report's gate discipline is the model: pick a numeric go/no-go per phase
(e.g. codec "within 2 dB PSNR of the reference codec") rather than chasing an absolute value.

### Go/no-go gates (score before spending on the next phase)

Adapted from the report's R3 rule ("numeric gates before the next phase's GPUs are rented").
Because RacerX has no published reference table, most gates are **trajectory/relative**, not absolute.

| phase | launcher | GO gate (proceed if…) | NO-GO (stop & fix if…) |
|---|---|---|---|
| **0. Data** | `test_mira_dataset` | PASS: frame counts match, action decoding matches source | any chunk-frame or action mismatch |
| **1. Smoke (all paths)** | `smoke_test_all` | every path writes a checkpoint; loss finite & trending down | NaN/Inf loss, no checkpoint, or a path errors |
| **2. Codec** | `train_codec` | recon loss descends; `latent_mean≈0`, `latent_std` stable (no collapse/explosion); (if comparing) PSNR within 2 dB of reference | loss flat/rising, latent std →0 or blows up |
| **3. World model** | `train`/`finetune` | val loss **monotone** ↓ with stable train-val gap; eval `drift`/`FDD`/`LPIPS` ↓ and `PSNR`/`SSIM` ↑ **vs the step-0 baseline** | val loss plateaus/rises early, or train↓ while eval↑ (overfit — need more clips) |
| **4. Playability (R5)** | hands-on (`run_mira`) | coherent world ≥1 s, objects persist, action response present | world dissolves, objects vanish, no steering response |

Two RacerX-specific notes:
- **Overfit is expected** at ~2 h of data — phase 3's NO-GO (train↓/eval↑) will trip until more
  clips encode. That's the gate telling you to wait for data, not a code bug.
- **Codec is usually the binding constraint** (report §4/§5): if WM eval saturates, improve/retrain
  the codec before scaling the world model.

## Data

RacerX playtest → mira WebDataset. Current state:

- All 663 clips have npz + physics + mira samples; **video mp4 encodes are the gate**
  (disk-bounded `stream_video`). Only clips with a `video_720p.mp4` become trainable latents.
- Full capture ≈ **42 h** of gameplay @ 20 fps (663 clips); trainable subset grows as encodes land.
- Build/refresh + validate the WebDataset: `<rx>\scripts\rebuild_dataset.bat`
- Index consumed by the trainer: `<rec>\mira_wds\train\index.json`

> Finetuning on a *few* clips overfits fast — wait for more of the pending mp4
> encodes before a real (non-smoke) run.
