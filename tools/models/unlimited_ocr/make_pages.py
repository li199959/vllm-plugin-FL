# Copyright (c) 2026 BAAI. All rights reserved.
"""Render N text pages with known ground truth for OCR accuracy checks.

  python3 make_pages.py <out_dir> [N=24]

Text comes from the model card README and LICENSE (real prose); each page varies
font size, line width and paragraph breaks. Writes page_XX.png and page_XX.txt.
"""

import random
import re
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

MODEL = Path("/mnt/cpfs/xiangbin/codex/Unlimited-OCR")


def corpus():
    text = (MODEL / "README.md").read_text() + "\n" + (MODEL / "LICENSE").read_text()
    text = re.sub(r"```.*?```", " ", text, flags=re.S)
    text = re.sub(r"<[^>]+>|!\[.*?\]\(.*?\)|\[([^\]]*)\]\([^)]*\)", r"\1", text)
    return [w for w in re.sub(r"[#*`|>-]+", " ", text).split() if w.isascii()]


def wrap(words, width):
    lines, cur = [], ""
    for w in words:
        if cur and len(cur) + len(w) + 1 > width:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    if cur:
        lines.append(cur)
    return lines


def main():
    out = Path(sys.argv[1])
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 24
    out.mkdir(parents=True, exist_ok=True)
    words = corpus()
    rng = random.Random(0)
    for i in range(n):
        size = rng.choice([16, 18, 20, 24, 28])
        width_chars = rng.choice([60, 75, 90])
        count = rng.randint(80, 600)
        start = rng.randrange(0, max(1, len(words) - count))
        chunk = words[start : start + count]
        lines = []
        for p in range(0, len(chunk), rng.randint(40, 120)):
            lines += wrap(chunk[p : p + 120], width_chars) + [""]
        font = ImageFont.load_default(size=size)
        page_w = int(width_chars * size * 0.6) + 100
        page_h = 80 + len(lines) * (size + 8)
        img = Image.new("RGB", (page_w, page_h), "white")
        draw = ImageDraw.Draw(img)
        y = 40
        for line in lines:
            draw.text((50, y), line, fill="black", font=font)
            y += size + 8
        img.save(out / f"page_{i:02d}.png")
        (out / f"page_{i:02d}.txt").write_text("\n".join(lines))
    print(f"wrote {n} pages to {out}")


if __name__ == "__main__":
    main()
