# Copyright (c) 2026 BAAI. All rights reserved.
"""Score OCR outputs against ground truth (word-level edit distance).

  python3 score_ocr.py <truth_dir> <out_dir_a> [<out_dir_b> ...]

For each page_XX.txt in truth_dir, reads page_XX.md from each output dir, strips
<|det|>/<|ref|> grounding markup and reports 1 - word_edit_distance / len(truth)
per page plus the mean, for each output dir.
"""

import re
import sys
from pathlib import Path


def words(text):
    text = re.sub(r"<\|det\|>.*?<\|/det\|>|<\|/?ref\|>", " ", text)
    return text.split()


def edit_distance(a, b):
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, y in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y))
        prev = cur
    return prev[-1]


def main():
    truth_dir, outs = Path(sys.argv[1]), [Path(p) for p in sys.argv[2:]]
    pages = sorted(truth_dir.glob("page_*.txt"))
    scores = {o: [] for o in outs}
    for t in pages:
        ref = words(t.read_text())
        row = [t.stem, str(len(ref))]
        for o in outs:
            f = o / (t.stem + ".md")
            hyp = words(f.read_text()) if f.exists() else []
            s = max(0.0, 1 - edit_distance(hyp, ref) / max(1, len(ref)))
            scores[o].append(s)
            row.append(f"{s:.4f}")
        print(" ".join(row))
    for o in outs:
        v = scores[o]
        exact = sum(s == 1.0 for s in v)
        print(
            f"{o}: mean {sum(v) / len(v):.4f}, exact {exact}/{len(v)}, min {min(v):.4f}"
        )


if __name__ == "__main__":
    main()
