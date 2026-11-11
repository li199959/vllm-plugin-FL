# Copyright (c) 2026 BAAI. All rights reserved.
"""Concurrent OCR throughput against a running Unlimited-OCR server.

  python3 bench_ocr.py --port 8120 --concurrency 16 --rounds 2 --out res.json <pages...>

Every round sends each page once (single-image requests, recipe sampling) with
``--concurrency`` requests in flight and reports requests/s, output tok/s and
latency. ``--profile`` wraps the last round in /start_profile and /stop_profile.
"""

import argparse
import base64
import json
import statistics
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def post(url, body, timeout=3600):
    req = urllib.request.Request(
        url, json.dumps(body).encode(), {"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r) if r.length != 0 else {}


def one(base, model, img_b64, max_tokens):
    body = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "skip_special_tokens": False,
        "vllm_xargs": {"ngram_size": 35, "window_size": 128},
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "<image>document parsing."},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{img_b64}"},
                    },
                ],
            }
        ],
    }
    t0 = time.time()
    d = post(f"{base}/v1/chat/completions", body)
    return (
        time.time() - t0,
        d["usage"]["prompt_tokens"],
        d["usage"]["completion_tokens"],
    )


def run_round(base, model, images, concurrency, max_tokens):
    t0 = time.time()
    with ThreadPoolExecutor(concurrency) as ex:
        res = list(ex.map(lambda b: one(base, model, b, max_tokens), images))
    wall = time.time() - t0
    lat = [r[0] for r in res]
    out = sum(r[2] for r in res)
    return {
        "requests": len(res),
        "wall_s": round(wall, 2),
        "req_per_s": round(len(res) / wall, 3),
        "out_tok_per_s": round(out / wall, 1),
        "prompt_tokens": sum(r[1] for r in res),
        "output_tokens": out,
        "mean_lat_s": round(statistics.mean(lat), 2),
        "p99_lat_s": round(sorted(lat)[max(0, int(len(lat) * 0.99) - 1)], 2),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--model", default="baidu/Unlimited-OCR")
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--out")
    ap.add_argument("images", nargs="+")
    args = ap.parse_args()
    base = f"http://127.0.0.1:{args.port}"
    images = [base64.b64encode(Path(p).read_bytes()).decode() for p in args.images]
    results = []
    for r in range(args.rounds):
        last = r == args.rounds - 1
        if args.profile and last:
            post(f"{base}/start_profile", {})
        try:
            res = run_round(base, args.model, images, args.concurrency, args.max_tokens)
        finally:
            if args.profile and last:
                post(f"{base}/stop_profile", {})
        res["round"] = r
        print(json.dumps(res), flush=True)
        results.append(res)
    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
