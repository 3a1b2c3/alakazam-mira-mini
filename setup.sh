#!/usr/bin/env bash
# Linux equivalent of setup.bat. On Linux the MIRA env is provided by PIXI (the
# repo ships a locked linux-64 pixi.lock), NOT the Windows uv/.venv+cu128 path --
# so setup.sh installs pixi and reproduces the exact locked env in the sibling
# mira repo (../mira). This is the same env train.sh / finetune.sh / smoke_*.sh
# run under (`pixi run --frozen`).
# Usage:  ./setup.sh
set -uo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
mira="$here/../mira"
[ -f "$mira/pixi.toml" ] || { echo "ERROR: mira trainer (with pixi.toml) not found at $mira -- clone it beside this repo"; exit 1; }

# ---- pixi -------------------------------------------------------------------
if ! command -v pixi >/dev/null 2>&1; then
    echo "--- installing pixi ---"
    curl -fsSL https://pixi.sh/install.sh | bash
fi
export PATH="$HOME/.pixi/bin:$PATH"

# ---- locked linux-64 env (solve + download once; no network at run time) ----
cd "$mira"
echo "--- pixi install --locked (conda base: python, ffmpeg, cuda libs) ---"
pixi install --locked || { echo "ERROR: pixi install failed"; exit 1; }

# ---- torch + torchcodec + mira (editable) INTO the env -----------------------
# pixi install only lays down the conda base; the Python ML deps (cu128 torch 2.8,
# torchcodec, and the editable mira package) come from the pixi `setup` task.
# Use `setup-cpu` on a machine with no NVIDIA GPU.
SETUP_TASK="${SETUP_TASK:-setup}"
echo "--- pixi run $SETUP_TASK (cu128 torch 2.8 + torchcodec + mira editable) ---"
pixi run "$SETUP_TASK" || { echo "ERROR: 'pixi run $SETUP_TASK' failed"; exit 1; }

# ---- tensorboard INTO the pixi env ------------------------------------------
# tracker.py does an UNGUARDED `from torch.utils.tensorboard import SummaryWriter`
# whenever tensorboard.logdir is set (which all training launchers now do by
# default) -- without tensorboard in THIS env the run CRASHES at the first log
# step, not just leaves an empty dir. Install it here (try uv pip -> pip -> pixi add).
echo "--- ensure tensorboard in the env (TB scalar logging) ---"
if pixi run --frozen python -c "import tensorboard" >/dev/null 2>&1; then
    echo "  tensorboard already present"
else
    pixi run --frozen uv pip install tensorboard >/dev/null 2>&1 \
     || pixi run --frozen python -m pip install tensorboard >/dev/null 2>&1 \
     || pixi add tensorboard >/dev/null 2>&1 \
     || echo "  WARN: could not install tensorboard into the env -- TB logging unavailable (wandb still works)"
    pixi run --frozen python -c "import tensorboard; print('  tensorboard', tensorboard.__version__)" 2>/dev/null \
     || echo "  tensorboard NOT importable in the env"
fi

# ---- DINOv3 hub repo (avoids the torch.hub GitHub 403 at train time) ---------
# The codec builds its DINOv3 backbone via torch.hub.load("facebookresearch/dinov3", ...), which
# pulls the repo's model DEFINITION from GitHub at model-construction time. On a shared cluster
# (horde) the anonymous GitHub API rate limit is quickly exhausted -> "HTTPError 403: rate limit
# exceeded" and training dies before step 0. Pre-clone the repo into the torch hub cache so
# torch.hub.load uses it locally and never hits GitHub. Dir MUST be <owner>_<repo>_<branch>.
hubdir="${TORCH_HOME:-$HOME/.cache/torch}/hub"
repodir="$hubdir/facebookresearch_dinov3_main"
if [ -d "$repodir" ]; then
    echo "--- dinov3 hub repo already cached: $repodir ---"
else
    echo "--- cloning facebookresearch/dinov3 -> $repodir (avoids torch.hub GitHub 403) ---"
    mkdir -p "$hubdir"
    git clone --depth 1 https://github.com/facebookresearch/dinov3 "$repodir" \
        || echo "WARN: dinov3 clone failed; torch.hub will fall back to GitHub (may 403 on a busy cluster)"
fi

# ---- verify -----------------------------------------------------------------
echo
echo "--- verify ---"
pixi run --frozen python -c "import torch, mira; print('torch', torch.__version__, 'cuda', torch.cuda.is_available()); print('mira ok')"
echo
echo "Setup complete: pixi env in $mira/.pixi"
echo "  dinov3 hub repo: $repodir $([ -d "$repodir" ] && echo OK || echo MISSING)"
echo "Next: ./download_models.sh  (rocket-science dataset)  or  ./download_weights.sh 1b"
