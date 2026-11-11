# Copyright (c) 2026 BAAI. All rights reserved.
"""OCR images through a running Unlimited-OCR server (vLLM recipe request format).

  python3 ocr_client.py --port 8100 --out <dir> img1.png [img2.png ...]

Each image is a single-image request (gundam/crop mode, window 128). Writes
<stem>.md (raw output, special tokens kept) and summary.json (tokens, seconds).
"""

import argparse
import base64
import json
import time
from pathlib import Path

from openai import OpenAI


def image_part(path):
    data = base64.b64encode(Path(path).read_bytes()).decode()
    return {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{data}"}}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--model", default="baidu/Unlimited-OCR")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--window", type=int, default=128)
    ap.add_argument("images", nargs="+")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    client = OpenAI(
        api_key="EMPTY", base_url=f"http://127.0.0.1:{args.port}/v1", timeout=3600
    )
    summary = {}
    for img in args.images:
        t0 = time.time()
        resp = client.chat.completions.create(
            model=args.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "<image>document parsing."},
                        image_part(img),
                    ],
                }
            ],
            max_tokens=args.max_tokens,
            temperature=0.0,
            extra_body={
                "skip_special_tokens": False,
                "vllm_xargs": {"ngram_size": 35, "window_size": args.window},
            },
        )
        text = resp.choices[0].message.content or ""
        (out / (Path(img).stem + ".md")).write_text(text)
        summary[Path(img).name] = {
            "prompt_tokens": resp.usage.prompt_tokens,
            "completion_tokens": resp.usage.completion_tokens,
            "finish_reason": resp.choices[0].finish_reason,
            "seconds": round(time.time() - t0, 2),
        }
        print(Path(img).name, summary[Path(img).name], flush=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
