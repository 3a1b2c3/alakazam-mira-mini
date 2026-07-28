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
echo "--- pixi install --locked (reproducing the locked linux-64 env) ---"
pixi install --locked || { echo "ERROR: pixi install failed"; exit 1; }

# ---- verify -----------------------------------------------------------------
echo
echo "--- verify ---"
pixi run --frozen python -c "import torch, mira; print('torch', torch.__version__, 'cuda', torch.cuda.is_available()); print('mira ok')"
echo
echo "Setup complete: pixi env in $mira/.pixi"
echo "Next: ./download_models.sh  (rocket-science dataset)  or  ./download_weights.sh 1b"
