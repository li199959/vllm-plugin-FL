#!/usr/bin/env python3
"""LibriSpeech ASR bench + WER for an OpenAI-compatible /v1/audio/transcriptions server.

Sends the local LibriSpeech flac files as-is (16 kHz mono, no client resampling),
with a fixed concurrency, and reports throughput (req/s, RTFx = audio sec / wall sec),
end-to-end latency percentiles and WER (Whisper EnglishTextNormalizer, as in the
official Whisper / Open ASR Leaderboard evaluation).

Usage:
  bench_asr.py --port 18360 --data <LibriSpeech/test-clean> [--num 0=all] [--concurrency 64]
               [--warmup 64] [--out result.json] [--profile]
"""

import argparse
import asyncio
import glob
import json
import os
import random
import time
from pathlib import Path

import aiohttp
import numpy as np
import soundfile as sf


def load_librispeech(root):
    items = []
    for trans in sorted(glob.glob(os.path.join(root, "*", "*", "*.trans.txt"))):
        d = os.path.dirname(trans)
        with open(trans) as fh:
            for line in fh:
                uid, text = line.strip().split(" ", 1)
                items.append((uid, os.path.join(d, uid + ".flac"), text))
    return items


def edit_distance(ref, hyp):
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i] + [0] * len(hyp)
        for j, h in enumerate(hyp, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h))
        prev = cur
    return prev[-1]


async def transcribe(session, url, model, path, audio_bytes, args):
    form = aiohttp.FormData()
    form.add_field(
        "file", audio_bytes, filename=os.path.basename(path), content_type="audio/flac"
    )
    form.add_field("model", model)
    form.add_field("language", args.language)
    form.add_field("temperature", "0")
    form.add_field("response_format", "json")
    t0 = time.perf_counter()
    async with session.post(url, data=form) as resp:
        body = await resp.text()
        dt = time.perf_counter() - t0
        if resp.status != 200:
            return None, dt, f"HTTP {resp.status}: {body[:200]}"
        return json.loads(body)["text"], dt, None


async def run(items, args):
    url = f"http://{args.host}:{args.port}/v1/audio/transcriptions"
    sem = asyncio.Semaphore(args.concurrency)
    payload = [
        (uid, p, Path(p).read_bytes(), ref, sf.info(p).duration)
        for uid, p, ref in items
    ]
    timeout = aiohttp.ClientTimeout(total=args.timeout)
    conn = aiohttp.TCPConnector(limit=args.concurrency)
    async with aiohttp.ClientSession(timeout=timeout, connector=conn) as session:

        async def one(item):
            uid, p, data, ref, dur = item
            async with sem:
                text, dt, err = await transcribe(
                    session, url, args.model, p, data, args
                )
            return dict(id=uid, ref=ref, hyp=text, latency=dt, duration=dur, error=err)

        if args.warmup:
            rng = random.Random(args.seed + 1)
            await asyncio.gather(
                *[one(x) for x in rng.sample(payload, min(args.warmup, len(payload)))]
            )
        base = f"http://{args.host}:{args.port}"
        if args.profile:
            async with session.post(f"{base}/start_profile") as r:
                assert r.status == 200, await r.text()
        t0 = time.perf_counter()
        results = await asyncio.gather(*[one(x) for x in payload])
        wall = time.perf_counter() - t0
        if args.profile:
            async with session.post(f"{base}/stop_profile") as r:
                assert r.status == 200, await r.text()
    return results, wall


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--model", default="whisper-large-v3")
    ap.add_argument(
        "--data",
        required=True,
        help="LibriSpeech split dir, e.g. .../LibriSpeech/test-clean",
    )
    ap.add_argument(
        "--normalizer", default="/mnt/cpfs/xiangbin/whisper-large-v3/normalizer.json"
    )
    ap.add_argument("--language", default="en")
    ap.add_argument("--num", type=int, default=0, help="0 = whole split")
    ap.add_argument("--concurrency", type=int, default=64)
    ap.add_argument("--warmup", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--timeout", type=float, default=3600)
    ap.add_argument("--out", default="")
    ap.add_argument(
        "--profile",
        action="store_true",
        help="wrap the timed run in /start_profile and /stop_profile (server needs --profiler-config)",
    )
    args = ap.parse_args()

    from transformers.models.whisper.english_normalizer import EnglishTextNormalizer

    with open(args.normalizer) as f:
        norm = EnglishTextNormalizer(json.load(f))

    items = load_librispeech(args.data)
    if args.num:
        items = random.Random(args.seed).sample(items, args.num)
    results, wall = asyncio.run(run(items, args))

    ok = [r for r in results if r["error"] is None]
    errs = sum(
        edit_distance(norm(r["ref"]).split(), norm(r["hyp"]).split()) for r in ok
    )
    words = sum(len(norm(r["ref"]).split()) for r in ok)
    lat = np.array([r["latency"] for r in ok]) * 1000 if ok else np.zeros(1)
    audio_s = sum(r["duration"] for r in ok)
    summary = dict(
        requests=len(results),
        failed=len(results) - len(ok),
        concurrency=args.concurrency,
        wall_s=round(wall, 3),
        req_per_s=round(len(ok) / wall, 2),
        audio_s=round(audio_s, 1),
        rtfx=round(audio_s / wall, 1),
        wer=round(100 * errs / max(words, 1), 3),
        words=words,
        lat_ms=dict(
            mean=round(float(lat.mean()), 1),
            p50=round(float(np.percentile(lat, 50)), 1),
            p95=round(float(np.percentile(lat, 95)), 1),
            p99=round(float(np.percentile(lat, 99)), 1),
        ),
    )
    print(json.dumps(summary, indent=1))
    for r in results:
        if r["error"]:
            print("ERROR", r["id"], r["error"])
            break
    if args.out:
        with open(args.out, "w") as f:
            json.dump(
                dict(summary=summary, results=results),
                f,
                indent=1,
                ensure_ascii=False,
            )


if __name__ == "__main__":
    main()
