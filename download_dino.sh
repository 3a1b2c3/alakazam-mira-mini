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

get() {   # $1=signed url  $2=target filename  $3=label
    local url="$1" fn="$2" label="$3"
    if [ -f "$DEST/$fn" ]; then echo "  have $label: $fn"; return; fi
    if [ -z "$url" ]; then echo "  SKIP $label: set its URL env var to download $fn"; return; fi
    echo "  downloading $label -> $fn"
    curl -L --fail -o "$DEST/$fn" "$url" || echo "  ERROR: download failed for $fn (check the signed URL/expiry)"
}

get "${DINOV3_VITL16_URL:-}" "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth" "vitl16 (large, codec training)"
get "${DINOV3_VITB16_URL:-}" "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth" "vitb16 (base, eval_wm metrics)"

echo
echo "dino_weights now holds:"
ls -1 "$DEST"/*.pth 2>/dev/null || true
