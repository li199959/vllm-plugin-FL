#!/usr/bin/env python3
"""GPQA Diamond prompt-protocol comparison against a running Qwen3-VL server.

Same questions and the same seeded option shuffle for every variant; only the
prompt (and for two variants the system prompt / greedy decoding) changes.
Sampling follows the Qwen3-VL tech report (arXiv 2511.21631) for Instruct
models: T=0.7, top_p=0.8, top_k=20, presence_penalty=1.5, max output 32768,
so the server needs --max-model-len >= ~33k. `boxed` reproduces the reported
70.4; EvalScope's "ANSWER: [LETTER]" template scores ~60. Writes one JSON
line per sample and prints accuracy, unanswered and truncated counts.

  python3 gpqa_prompt.py <port> <boxed|qwen_json|evalscope|...> out.jsonl [repeats=3]
"""

import json
import random
import re
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor

DATA = (
    "/mnt/cpfs/xiangbin/codex/profiles/qwen27b_eval_gpqa_erqa_20261006/datasets/"
    "datasets/AI-ModelScope--gpqa_diamond/snapshots/master/train.jsonl"
)

EVALSCOPE = (
    "Answer the following multiple choice question. The last line of your response "
    "should be of the following format: 'ANSWER: [LETTER]' (without quotes) where "
    "[LETTER] is one of A,B,C,D. Think step by step before answering.\n\n{q}\n\n{opts}"
)
SIMPLE_EVALS = (
    "Answer the following multiple choice question. The last line of your response "
    "should be of the following format: 'Answer: $LETTER' (without quotes) where "
    "LETTER is one of ABCD. Think step by step before answering.\n\n{q}\n\n{opts}"
)
# Qwen model-card recommendation for multiple-choice output standardisation.
QWEN_JSON = (
    "{q}\n\n{opts}\n\nPlease show your choice in the `answer` field with only the "
    'choice letter, e.g., `"answer": "C"`.'
)
BOXED = (
    "{q}\n\n{opts}\n\nPlease reason step by step, and put your final answer (the option "
    "letter) within \\boxed{{}}."
)

VARIANTS = {
    "evalscope": dict(tpl=EVALSCOPE),
    "simple_evals": dict(tpl=SIMPLE_EVALS),
    "qwen_json": dict(tpl=QWEN_JSON),
    "boxed": dict(tpl=BOXED),
    "sys_evalscope": dict(tpl=EVALSCOPE, system="You are a helpful assistant."),
    "greedy_evalscope": dict(tpl=EVALSCOPE, temperature=0.0, presence=0.0),
}


def load():
    items = []
    with open(DATA) as fh:
        for i, line in enumerate(fh):
            r = json.loads(line)
            opts = [
                r["Correct Answer"],
                r["Incorrect Answer 1"],
                r["Incorrect Answer 2"],
                r["Incorrect Answer 3"],
            ]
            opts = [o.strip() for o in opts]
            perm = list(range(4))
            random.Random(1000 + i).shuffle(perm)  # fixed per-question shuffle
            shuffled = [opts[p] for p in perm]
            target = "ABCD"[perm.index(0)]
            items.append((i, r["Question"].strip(), shuffled, target))
    return items


def extract(text):
    t = text.strip()
    # First matching pattern wins; within a pattern the last occurrence is taken.
    pats = [
        (r"ANSWER\s*[:：]\s*\(?\$?\**\s*([ABCD])\b", re.I),
        (r"\"answer\"\s*:\s*\"\s*([ABCD])", 0),
        (r"\\boxed\{\s*(?:\\text\{)?\(?([ABCD])\b", 0),
        (r"[Aa]nswer\s*(?:is)?\s*[:：]?\s*\**\(?([ABCD])\b", 0),
        # JSON field written as `answer`: "A" / answer": "A" (common with qwen_json)
        (r"answer[`\"'\s]*:\s*[`\"'\s]*\(?([ABCD])\b", re.I),
    ]
    for p, flags in pats:
        m = re.findall(p, t, flags=flags)
        if m:
            return m[-1].upper()
    return None


def call(port, msgs, cfg, seed):
    body = dict(
        model="qwen3-vl",
        messages=msgs,
        max_tokens=32768,
        temperature=cfg.get("temperature", 0.7),
        top_p=0.8,
        top_k=20,
        presence_penalty=cfg.get("presence", 1.5),
        seed=seed,
    )
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=3600) as f:
                r = json.load(f)
            c = r["choices"][0]
            return (
                c["message"]["content"] or "",
                c["finish_reason"],
                r["usage"]["completion_tokens"],
            )
        except Exception as e:  # retry transient errors
            err = e
    raise err


def main():
    port, name, out = sys.argv[1], sys.argv[2], sys.argv[3]
    repeats = int(sys.argv[4]) if len(sys.argv) > 4 else 3
    cfg = VARIANTS[name]
    jobs = [(it, k) for k in range(repeats) for it in load()]

    def run(job):
        (i, q, opts, target), k = job
        o = "\n".join(f"{'ABCD'[j]}) {opts[j]}" for j in range(4))
        msgs = [{"role": "user", "content": cfg["tpl"].format(q=q, opts=o)}]
        if "system" in cfg:
            msgs.insert(0, {"role": "system", "content": cfg["system"]})
        text, fin, ntok = call(port, msgs, cfg, seed=42 + k)
        pred = extract(text)
        return dict(
            idx=i,
            rep=k,
            target=target,
            pred=pred,
            correct=pred == target,
            finish=fin,
            tokens=ntok,
            response=text,
        )

    with ThreadPoolExecutor(64) as ex, open(out, "w") as f:
        res = []
        for r in ex.map(run, jobs):
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            res.append(r)
    n = len(res)
    acc = sum(r["correct"] for r in res) / n * 100
    none = sum(r["pred"] is None for r in res)
    trunc = sum(r["finish"] == "length" for r in res)
    med = sorted(r["tokens"] for r in res)[n // 2]
    print(
        f"RESULT {name} n={n} acc={acc:.2f} no_answer={none} truncated={trunc} median_tok={med}"
    )


if __name__ == "__main__":
    main()
