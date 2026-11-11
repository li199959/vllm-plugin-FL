#!/usr/bin/env python3
"""Quick vision sanity check against running Qwen3-VL servers.

Solid colors at 448 px and the first ERQA images (robot scenes) with their
questions; prints the model's answer next to the ground truth. Native and
optimized servers should agree with the image content.

  python3 vision_check.py 9501 9503 [--n 8]
"""

import argparse
import base64
import io
import json
import urllib.request

import pyarrow.parquet as pq
from PIL import Image

ERQA = (
    "/mnt/cpfs/xiangbin/codex/profiles/qwen27b_eval_gpqa_erqa_20261006/datasets/datasets/"
    "evalscope--ERQA/snapshots/master/data/test-00000-of-00001.parquet"
)
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def ask(port, images, text, max_tokens=64):
    content = []
    for img in images:
        b = io.BytesIO()
        img.save(b, format="PNG")
        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": "data:image/png;base64,"
                    + base64.b64encode(b.getvalue()).decode()
                },
            }
        )
    content.append({"type": "text", "text": text})
    body = {
        "model": "qwen3-vl",
        "messages": [{"role": "user", "content": content}],
        "temperature": 0,
        "max_tokens": max_tokens,
    }
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    return json.load(OPENER.open(req, timeout=600))["choices"][0]["message"][
        "content"
    ].strip()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("ports", type=int, nargs="+")
    ap.add_argument("--n", type=int, default=8)
    a = ap.parse_args()
    rows = pq.read_table(ERQA).slice(0, a.n).to_pylist()
    for port in a.ports:
        print(f"== port {port}")
        for color, rgb in (("red", (255, 0, 0)), ("blue", (0, 0, 255))):
            ans = ask(
                port,
                [Image.new("RGB", (448, 448), rgb)],
                "What is the color of this image? Answer with one color word.",
                8,
            )
            print(f"  color {color}: {ans!r}")
        ans = ask(
            port,
            [Image.open(io.BytesIO(rows[0]["images"][0]["bytes"])).convert("RGB")],
            "Describe this image in one sentence.",
        )
        print(f"  describe erqa[0]: {ans!r}")
        correct = 0
        for r in rows:
            imgs = [
                Image.open(io.BytesIO(i["bytes"])).convert("RGB") for i in r["images"]
            ]
            ans = ask(port, imgs, r["question"], 8)
            correct += ans[:1].upper() == r["answer"].strip()[:1].upper()
            print(f"  {r['question_id']}: gt={r['answer']} got={ans!r}")
        print(f"  first {a.n}: {correct}/{a.n}")


if __name__ == "__main__":
    main()
