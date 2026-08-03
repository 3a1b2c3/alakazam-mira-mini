#!/usr/bin/env python3
"""Codec reconstruction PSNR on RacerX clips -- isolates STAGE 1 (no world model).

Encodes each clip through the frozen mira-mini codec and decodes it straight back
(encode->decode), then PSNR(input, reconstruction). This is the *ceiling* on video
quality: the world model can never produce better RacerX pixels than the codec can
reconstruct. Low here => the (Rocket League) codec is the domain-transfer bottleneck.

Run with the mira venv (has `mira` + torchcodec):
  ..\mira\.venv\Scripts\python.exe codec_recon_psnr.py --data C:\recordings\mira_wds\train\index.json --num 8
"""
import argparse
import glob
import json
import math
import os
import tarfile
import tempfile
from pathlib import Path

# The codec torch.compile's the DINO forward; inductor fails on CPU (and needs a
# compiler on Windows). Neuter compile so encode/decode run eager. Must be set
# BEFORE importing torch (dynamo reads these at import).
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")

import torch


def _no_compile(model=None, **_kw):
    return (lambda fn: fn) if model is None else model


torch.compile = _no_compile  # type: ignore[assignment]

import torch.nn.functional as F
from einops import rearrange
from torchcodec.decoders import VideoDecoder

from mira.codec.codec_model import VideoCodec

# torch.hub loads DINOv3 with source="github", which pings GitHub for the ref even
# when the repo is already cached -- that fails behind a firewall (RemoteDisconnected).
# Redirect the cached repo to source="local" so the codec's DINO loads OFFLINE.
_ORIG_HUB_LOAD = torch.hub.load


def _offline_hub_load(repo_or_dir, *a, **kw):
    cached = Path.home() / ".cache" / "torch" / "hub" / "facebookresearch_dinov3_main"
    if isinstance(repo_or_dir, str) and "dinov3" in repo_or_dir.lower() and cached.is_dir():
        kw["source"] = "local"
        return _ORIG_HUB_LOAD(str(cached), *a, **kw)
    return _ORIG_HUB_LOAD(repo_or_dir, *a, **kw)


torch.hub.load = _offline_hub_load


def _default_codec() -> str:
    hits = sorted(glob.glob(
        r"C:\Users\kschmid\.cache\huggingface\hub\models--alakazamworld--mira-mini"
        r"\snapshots\*\codec\checkpoint-125000\checkpoint.pth"))
    return hits[-1] if hits else ""


def _decode_clip(shard: Path, member: str) -> torch.Tensor:
    """One clip's .p0.mp4 -> (T, C, H, W) uint8."""
    with tarfile.open(shard) as tar:
        data = tar.extractfile(member).read()
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp.write(data)
        path = tmp.name
    try:
        dec = VideoDecoder(path)
        frames = torch.stack([f.cpu() for f in dec])  # (T, C, H, W) uint8
    finally:
        try:
            Path(path).unlink()
        except OSError:
            pass
    return frames


def main() -> int:
    ap = argparse.ArgumentParser(description="Codec encode->decode reconstruction PSNR on RacerX")
    ap.add_argument("--data", required=True, help="path to a mira_wds index.json")
    ap.add_argument("--codec", default=_default_codec(), help="codec checkpoint.pth (default: mira-mini 125k)")
    ap.add_argument("--num", type=int, default=8, help="number of clips to evaluate")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    idx = json.load(open(args.data, encoding="utf-8"))
    root = Path(args.data).parent
    entries = idx["entries"][: args.num]
    if not args.codec or not Path(args.codec).is_file():
        print("ERROR: codec checkpoint not found; pass --codec"); return 1

    print(f"codec: {args.codec}")
    codec = VideoCodec.load_from_checkpoint(args.codec, device=args.device).eval()
    tgt_h, tgt_w = codec.config.encoder.video.height, codec.config.encoder.video.width
    print(f"codec input res: {tgt_h}x{tgt_w} | clips: {len(entries)}\n")

    psnrs = []
    for e in entries:
        shard = root / e["shard"]
        with tarfile.open(shard) as tar:
            member = next((m.name for m in tar.getmembers()
                           if e["match_id"] in m.name and m.name.endswith(".p0.mp4")), None)
        if member is None:
            continue
        frames = _decode_clip(shard, member)                       # (T,C,H,W) uint8
        video = (frames.float() / 255.0).unsqueeze(0).to(args.device)  # (1,T,C,H,W) [0,1]
        # resize to codec input (mirror preprocess_batch)
        b, t = video.shape[:2]
        video = F.interpolate(rearrange(video, "b t c h w -> (b t) c h w"),
                              size=(tgt_h, tgt_w), mode="bilinear", align_corners=False)
        video = rearrange(video, "(b t) c h w -> b t c h w", b=b, t=t)
        with torch.no_grad():
            input_video, enc = codec.encode(video)                 # normalizes to [-1,1], trims T
            recon = codec.decode(enc.z)
        # both back to [0,1]; align T
        tt = min(input_video.shape[1], recon.shape[1])
        inp01 = (input_video[:, :tt] * 0.5 + 0.5).clamp(0, 1)
        rec01 = (recon[:, :tt] * 0.5 + 0.5).clamp(0, 1)
        mse = F.mse_loss(rec01, inp01).item()
        psnr = -10.0 * math.log10(mse + 1e-12)
        psnrs.append(psnr)
        print(f"  {e['match_id'][:24]:24s}  PSNR {psnr:5.2f} dB")

    if psnrs:
        mean = sum(psnrs) / len(psnrs)
        print(f"\n=== codec reconstruction PSNR (RacerX, stage-1 only) ===")
        print(f"  mean {mean:.2f} dB over {len(psnrs)} clips  (target ~28.6)")
        print("  << ~28 -> the RL codec can't hold RacerX -> stage-1 IS your transfer bottleneck")
        print("  ~ target -> codec is fine; look at the WM finetune / data instead")
    else:
        print("no clips evaluated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
