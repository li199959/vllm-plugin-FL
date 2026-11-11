#!/usr/bin/env python3
"""Numerical check of the serving stack: teacher-forced per-token logprobs of
fixed GPQA answer texts, HF transformers (plain torch eager, reference) vs
vLLM servers (prompt_logprobs on /v1/completions with the identical token ids).

Reports per server: mean |dlogprob| over answer tokens, fraction of tokens
whose reference argmax differs, and the total sequence log-likelihood gap.

  python3 logprob_check.py --hf-gpus 6,7 --ports 9801 9802 --n 6
"""

import argparse
import json
import os
import urllib.request

MODEL = "/mnt/cpfs/xiangbin/Qwen3-VL-30B-A3B-Instruct"
PRED = (
    "/mnt/cpfs/xiangbin/codex/profiles/q_eval_native_gpqa_20261020/results/predictions/"
    "Qwen3-VL-30B-A3B-FL/gpqa_diamond_default.jsonl"
)
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def samples(n, max_answer_tokens, tok):
    out = []
    with open(PRED) as fh:
        for line in fh:
            r = json.loads(line)
            q = r["messages"][0]["content"]
            q = (
                q
                if isinstance(q, str)
                else " ".join(x.get("text", "") for x in q if isinstance(x, dict))
            )
            a = r["model_output"]["choices"][0]["message"]["content"]
            # transformers 5.x returns a BatchEncoding for tokenize=True; render text first.
            text = tok.apply_chat_template(
                [{"role": "user", "content": q}],
                add_generation_prompt=True,
                tokenize=False,
            )
            prompt_ids = tok(text, add_special_tokens=False)["input_ids"]
            ans_ids = tok(a, add_special_tokens=False)["input_ids"][:max_answer_tokens]
            out.append((list(prompt_ids), list(ans_ids)))
            if len(out) == n:
                break
    return out


def vllm_logprobs(port, ids):
    body = {
        "model": "qwen3-vl",
        "prompt": ids,
        "max_tokens": 1,
        "temperature": 0,
        "prompt_logprobs": 1,
    }
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    r = json.load(OPENER.open(req, timeout=900))
    plp = r["choices"][0][
        "prompt_logprobs"
    ]  # list (None for first) of {tok_id: {logprob, rank}}
    lp, top = [], []
    for i, d in enumerate(plp):
        if d is None:
            lp.append(None)
            top.append(None)
            continue
        own = d.get(str(ids[i]))
        lp.append(own["logprob"] if own else float("-inf"))
        best = min(d.items(), key=lambda kv: kv[1].get("rank", 1e9))
        top.append(int(best[0]))
    return lp, top


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hf-gpus", default="6,7")
    ap.add_argument("--ports", type=int, nargs="+", required=True)
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--answer-tokens", type=int, default=400)
    a = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = a.hf_gpus
    import torch
    from transformers import AutoTokenizer, Qwen3VLMoeForConditionalGeneration

    tok = AutoTokenizer.from_pretrained(MODEL)
    data = samples(a.n, a.answer_tokens, tok)
    model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map="auto", attn_implementation="eager"
    ).eval()
    stats = {p: {"abs": [], "flip": 0, "tok": 0, "ll": 0.0} for p in a.ports}
    ref_ll = 0.0
    for k, (pids, aids) in enumerate(data):
        ids = pids + aids
        with torch.inference_mode():
            logits = (
                model(input_ids=torch.tensor([ids], device="cuda:0")).logits[0].float()
            )
        logp = torch.log_softmax(logits, -1)
        start = len(pids)
        ref = [logp[i - 1, ids[i]].item() for i in range(start, len(ids))]
        ref_top = [int(logp[i - 1].argmax()) for i in range(start, len(ids))]
        ref_ll += sum(ref)
        for p in a.ports:
            lp, top = vllm_logprobs(p, ids)
            s = stats[p]
            for j, i in enumerate(range(start, len(ids))):
                s["abs"].append(abs(lp[i] - ref[j]))
                s["flip"] += top[i] != ref_top[j]
                s["tok"] += 1
                s["ll"] += lp[i]
        print(f"sample {k}: prompt {len(pids)} answer {len(aids)} tokens", flush=True)
    for p, s in stats.items():
        absd = sorted(s["abs"])
        print(
            f"port {p}: tokens {s['tok']} mean|dlogp| {sum(absd) / len(absd):.4f} "
            f"p99 {absd[int(0.99 * len(absd))]:.3f} max {absd[-1]:.3f} "
            f"argmax_flips {s['flip']} ({100 * s['flip'] / s['tok']:.2f}%) "
            f"loglik gap {s['ll'] - ref_ll:+.2f} (ref {ref_ll:.2f})"
        )


if __name__ == "__main__":
    main()
