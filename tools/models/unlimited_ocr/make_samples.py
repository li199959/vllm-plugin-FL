# Copyright (c) 2026 BAAI. All rights reserved.
"""Build OCR sample images with known text (PIL's bundled font; no system fonts here).

  python3 make_samples.py <out_dir>

Writes doc_short.png (one paragraph + a small table), doc_long.png (a dense page
of real prose, crosses the 128-token R-SWA window many times), their ground-truth
.txt files, and frames from the model card's long-horizon-ocr.gif.
"""

import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

MODEL = Path("/mnt/cpfs/xiangbin/codex/Unlimited-OCR")


def render(lines, path, width=1240, size=22, gap=10):
    font = ImageFont.load_default(size=size)
    height = 80 + len(lines) * (size + gap)
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    y = 40
    for line in lines:
        draw.text((50, y), line, fill="black", font=font)
        y += size + gap
    img.save(path)
    return img.size


def wrap(words, width=90):
    lines, cur = [], ""
    for w in words:
        if len(cur) + len(w) + 1 > width:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    if cur:
        lines.append(cur)
    return lines


def main():
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    meta = {}

    short = [
        "Unlimited OCR Works: One-shot Long-horizon Parsing",
        "",
        "The model reads a full page and writes markdown in a single pass.",
        "Prompt and image tokens stay visible while generated tokens use a window.",
        "",
        "| Model | Pages | Score |",
        "| Base | 1 | 87.5 |",
        "| Gundam | 32 | 91.2 |",
    ]
    meta["doc_short.png"] = render(short, out / "doc_short.png")
    (out / "doc_short.txt").write_text("\n".join(short))

    # Real prose (the model card's README text, markup stripped) wrapped into a
    # dense page: ~1.2k words, far past the 128-token R-SWA window.
    import re

    readme = (MODEL / "README.md").read_text() + "\n" + (MODEL / "LICENSE").read_text()
    readme = re.sub(r"```.*?```", " ", readme, flags=re.S)
    readme = re.sub(r"<[^>]+>|!\[.*?\]\(.*?\)|\[([^\]]*)\]\([^)]*\)", r"\1", readme)
    words = re.sub(r"[#*`|>-]+", " ", readme).split()
    words = [w for w in words if w.isascii()]
    paras = [words[i : i + 110] for i in range(0, min(len(words), 1200), 110)]
    long_lines = []
    for p in paras:
        long_lines += wrap(p) + [""]
    meta["doc_long.png"] = render(long_lines, out / "doc_long.png", size=18, gap=6)
    (out / "doc_long.txt").write_text("\n".join(long_lines))

    gif = Image.open(MODEL / "assets/long-horizon-ocr.gif")
    for k in (0, gif.n_frames // 2, gif.n_frames - 1):
        gif.seek(k)
        gif.convert("RGB").save(out / f"gif_frame_{k:03d}.png")
        meta[f"gif_frame_{k:03d}.png"] = gif.size
    for name in ("Unlimited-OCR.png", "baidu.png"):
        Image.open(MODEL / "assets" / name).convert("RGB").save(out / name)
    print(json.dumps(meta))


if __name__ == "__main__":
    main()
