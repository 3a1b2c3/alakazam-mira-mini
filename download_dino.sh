#!/usr/bin/env bash
# Linux mirror of download_dino.bat. Download the (Meta-gated) DINOv3 backbone
# weights into the sibling mira repo's dino_weights/, used by codec TRAINING
# (vitl16, large) and the world-model FDD METRIC eval_wm.sh runs (vitb16, base).
# The weights are gated, so supply your own time-limited SIGNED download URLs via
# env vars (get them from
#   https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/):
#
#   export DINOV3_VITL16_URL=<signed url for the large weights>
#   export DINOV3_VITB16_URL=<signed url for the base  weights>
#   ./download_dino.sh
#
# Skips any file already present, so re-run after adding a URL.
set -uo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
mira="$here/../mira"
DEST="${DINO_DEST:-$mira/dino_weights}"
mkdir -p "$DEST"

get() {   # $1=env var name  $2=default URL  $3=target filename  $4=label  $5=hf_repo_id  $6=hf_filename
    local env_name="$1" default_url="$2" fn="$3" label="$4" hf_repo="$5" hf_fn="$6"
    if [ -f "$DEST/$fn" ]; then echo "  have $label: $fn"; return; fi
    local url="${!env_name:-$default_url}"
    if [ -z "$url" ]; then echo "  SKIP $label: set $env_name to download $fn"; return; fi

    echo "  downloading $label -> $fn"
    # Try curl first
    if curl -L --fail -o "$DEST/$fn" "$url" 2>/dev/null; then
        echo "    ✓ downloaded via curl"
        return
    fi

    # Fallback: use huggingface_hub Python library (handles auth automatically)
    echo "    curl failed, trying huggingface_hub..."

    # Check if HF_TOKEN is set
    if [ -z "${HF_TOKEN:-}" ]; then
        echo "  ✗ ERROR: HF_TOKEN not set. Set with:"
        echo "      export HF_TOKEN='hf_your_token_here'"
        return 1
    fi

    # Try Python fallback
    if command -v python3 &>/dev/null; then
        python3 << PYTHON_EOF
import os
from huggingface_hub import hf_hub_download
token = os.environ.get("HF_TOKEN")
if not token:
    print("  ✗ ERROR: HF_TOKEN not set")
    exit(1)
try:
    path = hf_hub_download(
        repo_id="$hf_repo",
        filename="$hf_fn",
        cache_dir="$DEST",
        token=token,
    )
    print(f"    ✓ downloaded via huggingface_hub to {path}")
except Exception as e:
    print(f"  ✗ ERROR: huggingface_hub failed: {e}")
    exit(1)
PYTHON_EOF
        return $?
    else
        echo "  ✗ ERROR: python3 not available (need curl OR python3)"
        return 1
    fi
}

# Default URLs from HuggingFace
DINOV3_VITL16_DEFAULT="https://huggingface.co/facebookresearch/dinov3/resolve/main/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
DINOV3_VITB16_DEFAULT="https://huggingface.co/facebookresearch/dinov3/resolve/main/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"

get "DINOV3_VITL16_URL" "$DINOV3_VITL16_DEFAULT" "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth" "vitl16 (large, codec training)" "facebookresearch/dinov3" "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
get "DINOV3_VITB16_URL" "$DINOV3_VITB16_DEFAULT" "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth" "vitb16 (base, eval_wm metrics)" "facebookresearch/dinov3" "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"

# torch.hub loads the DINOv3 model DEFINITION (not the weights) from the facebookresearch/dinov3
# GitHub repo at model-construction time. On a shared cluster (horde), many anonymous nodes exhaust
# GitHub's API rate limit -> "HTTPError 403: rate limit exceeded" and training dies before step 0.
# Pre-clone the repo into the torch hub cache so torch.hub.load finds it locally ("Using cache found
# in ...") and never touches GitHub. Dir name MUST be <owner>_<repo>_<branch> = facebookresearch_dinov3_main.
hubdir="${TORCH_HOME:-$HOME/.cache/torch}/hub"
repodir="$hubdir/facebookresearch_dinov3_main"
if [ -d "$repodir" ]; then
    echo "  have dinov3 hub repo: $repodir"
else
    echo "  cloning facebookresearch/dinov3 -> $repodir (avoids torch.hub GitHub rate limit)"
    mkdir -p "$hubdir"
    git clone --depth 1 https://github.com/facebookresearch/dinov3 "$repodir" \
        || echo "  ERROR: git clone dinov3 failed; torch.hub will fall back to GitHub (may 403 on a busy cluster)"
fi

echo
echo "dino_weights now holds:"
ls -1 "$DEST"/*.pth 2>/dev/null || true
echo "torch hub repo: $repodir $([ -d "$repodir" ] && echo OK || echo MISSING)"
