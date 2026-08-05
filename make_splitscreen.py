#!/usr/bin/env python3
"""Make split-screen comparison images: LEFT = original (input), RIGHT = encoded (recon).

Pairs every ``*_input.png`` with its ``*_recon.png`` sibling in a directory and
writes ``*_split.png`` (original | encoded, labelled). Default dir is the frozen
codec eval output.

Usage:
  python make_splitscreen.py [DIR]
  python make_splitscreen.py C:\\workspace\\world\\alakazam-mira-mini\\outputs\\frozen_codec_eval
"""
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

DEFAULT_DIR = Path(r"C:\workspace\world\alakazam-mira-mini\outputs\frozen_codec_eval")
DIVIDER = 3            # px black divider between the two halves
LABEL_H = 28          # px label strip height
BG = (0, 0, 0)


def _font(size: int = 20) -> ImageFont.ImageFont:
    for name in ("arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _label(draw: ImageDraw.ImageDraw, x: int, w: int, text: str, font) -> None:
    tw = draw.textlength(text, font=font)
    draw.text((x + (w - tw) / 2, 4), text, fill=(255, 255, 255), font=font)


def make_split(inp: Path, rec: Path, out: Path, font) -> None:
    a = Image.open(inp).convert("RGB")
    b = Image.open(rec).convert("RGB")
    # match heights (recon may differ slightly)
    if a.size != b.size:
        b = b.resize(a.size, Image.Resampling.LANCZOS)
    w, h = a.size
    canvas = Image.new("RGB", (w * 2 + DIVIDER, h + LABEL_H), BG)
    canvas.paste(a, (0, LABEL_H))
    canvas.paste(b, (w + DIVIDER, LABEL_H))
    draw = ImageDraw.Draw(canvas)
    _label(draw, 0, w, "ORIGINAL", font)
    _label(draw, w + DIVIDER, w, "ENCODED", font)
    canvas.save(out)
    print(f"  wrote {out}")


def main() -> int:
    d = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_DIR
    if not d.is_dir():
        print(f"ERROR: dir not found: {d}")
        return 1
    font = _font(20)
    inputs = sorted(d.glob("*_input.png"))
    if not inputs:
        print(f"no *_input.png in {d}")
        return 1
    n = 0
    for inp in inputs:
        rec = inp.with_name(inp.name.replace("_input.png", "_recon.png"))
        if not rec.is_file():
            print(f"  skip {inp.name}: no matching _recon.png")
            continue
        out = inp.with_name(inp.name.replace("_input.png", "_split.png"))
        make_split(inp, rec, out, font)
        n += 1
    print(f"\n{n} split-screen image(s) written to {d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
