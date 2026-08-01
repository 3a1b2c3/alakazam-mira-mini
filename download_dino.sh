#!/usr/bin/env bash
# Linux mirror of download_dino.bat. Download DINOv3 backbone weights from HuggingFace
# into the sibling mira repo's dino_weights/, used by codec TRAINING (vitl16, large) and
# the world-model FDD METRIC eval_wm.sh runs (vitb16, base).
#
# Default: downloads from HuggingFace (facebookresearch/dinov3).
# Override with env vars if needed:
#   export DINOV3_VITL16_URL=<custom url for the large weights>
#   export DINOV3_VITB16_URL=<custom url for the base  weights>
#   ./download_dino.sh
#
# Skips any file already present, so re-run to retry failed downloads.
set -uo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
mira="$here/../mira"
DEST="${DINO_DEST:-$mira/dino_weights}"
mkdir -p "$DEST"

get() {   # $1=env var name  $2=default URL  $3=target filename  $4=label  $5=hf_repo_id  $6=hf_filename
    local env_name="$1" default_url="$2" fn="$3" label="$4" hf_repo="$5" hf_fn="$6"
    if [ -f "$DEST/$fn" ]; then echo "  have $label: $fn"; return; fi
    local url="${!env_name:-$default_url}"
    if [ -z "$url" ]; then echo "  SKIP $label: no URL set"; return; fi

    echo "  downloading $label -> $fn"
    # Try curl first
    if curl -L --fail -o "$DEST/$fn" "$url" 2>/dev/null; then
        echo "    ✓ downloaded via curl"
        return
    fi

    # Fallback: use huggingface_hub Python library (handles auth automatically)
    echo "    curl failed, trying huggingface_hub..."

    # Try Python fallback
    if command -v python3 &>/dev/null; then
        python3 << PYTHON_EOF
import os
from huggingface_hub import hf_hub_download
token = os.environ.get("HF_TOKEN")
try:
    path = hf_hub_download(
        repo_id="$hf_repo",
        filename="$hf_fn",
        cache_dir="$DEST",
        token=token,
    )
    print(f"    ✓ downloaded via huggingface_hub")
except Exception as e:
    print(f"  ✗ ERROR: download failed: {e}")
    exit(1)
PYTHON_EOF
        return $?
    else
        echo "  ✗ ERROR: python3 not available (need curl OR python3)"
        return 1
    fi
}

# Meta's CDN signed URLs (time-limited, from https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/)
DINOV3_VITL16_DEFAULT="https://dinov3.llamameta.net/dinov3_vitl16/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth?Policy=eyJTdGF0ZW1lbnQiOlt7InVuaXF1ZV9oYXNoIjoibHE1Mm8za3MxcmhyYnhtYzlyNG1qM2RxIiwiUmVzb3VyY2UiOiJodHRwczpcL1wvZGlub3YzLmxsYW1hbWV0YS5uZXRcLyoiLCJDb25kaXRpb24iOnsiRGF0ZUxlc3NUaGFuIjp7IkFXUzpFcG9jaFRpbWUiOjE3ODU3NTYwNzl9fX1dfQ__&Signature=qEQg0t3HUKGSg5B5%7EnRVsOgSmrDjzWZNK2ERy10NInTlUktcJFnoarBmPINuZzoYJvaptmhnkndOB9RaeNh%7EXHnkxNRpL5t17KQkhvGjoJNS6WcmJUB4AZKuvxFWQxY25vYwRBfaMRumu-VYLDGbMpMHoWB0twn9TsNl4k7oLjWyVqbqyhZAgE5JGULBj5O2USHmfqCX38RRvEJQjXXX%7EfsH0oMxt8VlQw5qWMEUdBSchTYATnr7XlNnlozchrZ2tM7Z8ruGiAg5p4ZixIhK9IAnoB3YCUEM%7EgImpXIsPrQBom8ylyU5k3BMBnKSf-f5465YVpWloZgt3J1tihnuYQ__&Key-Pair-Id=K15QRJLYKIFSLZ&Download-Request-ID=1712895043303984"
DINOV3_VITB16_DEFAULT="https://dinov3.llamameta.net/dinov3_vitb16/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth?Policy=eyJTdGF0ZW1lbnQiOlt7InVuaXF1ZV9oYXNoIjoibHE1Mm8za3MxcmhyYnhtYzlyNG1qM2RxIiwiUmVzb3VyY2UiOiJodHRwczpcL1wvZGlub3YzLmxsYW1hbWV0YS5uZXRcLyoiLCJDb25kaXRpb24iOnsiRGF0ZUxlc3NUaGFuIjp7IkFXUzpFcG9jaFRpbWUiOjE3ODU3NTYwNzl9fX1dfQ__&Signature=qEQg0t3HUKGSg5B5%7EnRVsOgSmrDjzWZNK2ERy10NInTlUktcJFnoarBmPINuZzoYJvaptmhnkndOB9RaeNh%7EXHnkxNRpL5t17KQkhvGjoJNS6WcmJUB4AZKuvxFWQxY25vYwRBfaMRumu-VYLDGbMpMHoWB0twn9TsNl4k7oLjWyVqbqyhZAgE5JGULBj5O2USHmfqCX38RRvEJQjXXX%7EfsH0oMxt8VlQw5qWMEUdBSchTYATnr7XlNnlozchrZ2tM7Z8ruGiAg5p4ZixIhK9IAnoB3YCUEM%7EgImpXIsPrQBom8ylyU5k3BMBnKSf-f5465YVpWloZgt3J1tihnuYQ__&Key-Pair-Id=K15QRJLYKIFSLZ&Download-Request-ID=1712895043303984"

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
