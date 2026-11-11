# Copyright (c) 2026 BAAI. All rights reserved.
"""Compare a bge-m3 server's dense embeddings with an fp32 HF reference.

The reference is XLMRobertaModel -> CLS token -> L2 normalize, which is what the
model's sentence-transformers config (1_Pooling: cls, 2_Normalize) computes.
Reports per-text cosine to the reference and whether the query->passage ranking
of a small multilingual retrieval set matches.

  python3 check_embeddings.py --ref ref.npz                    # build reference (CPU ok)
  python3 check_embeddings.py --ref ref.npz --port 8000 --out check.json
"""

import argparse
import json
import urllib.request

import numpy as np

MODEL = "/mnt/cpfs/xiangbin/codex/bge-m3"

QUERIES = [
    "What is BGE M3?",
    "Defination of BM25",
    "什么是大熊猫？",
    "Wie hoch ist der Mount Everest?",
    "東京の人口は？",
]
PASSAGES = [
    "BGE M3 is an embedding model supporting dense retrieval, lexical matching and "
    "multi-vector interaction.",
    "BM25 is a bag-of-words retrieval function that ranks a set of documents based on the "
    "query terms appearing in each document",
    "大熊猫是一种熊科哺乳动物，主要栖息在中国四川、陕西和甘肃的山区，以竹子为主要食物。",
    "Der Mount Everest ist mit 8848 Metern der höchste Berg der Erde.",
    "東京都の人口は約1400万人で、日本の都道府県の中で最も多い。",
]


def long_text(n_words):
    base = (
        "Retrieval augmented generation combines a retriever with a generator so that "
        "answers are grounded in documents. "
    )
    return (base * (n_words // 16 + 1))[: n_words * 6]


TEXTS = QUERIES + PASSAGES + [long_text(400), long_text(3000)]


def reference(device):
    import torch
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = (
        AutoModel.from_pretrained(MODEL, torch_dtype=torch.float32).to(device).eval()
    )
    out = []
    with torch.no_grad():
        for t in TEXTS:
            x = tok(t, return_tensors="pt", truncation=True, max_length=8192).to(device)
            cls = model(**x).last_hidden_state[0, 0]
            out.append(torch.nn.functional.normalize(cls, dim=-1).cpu().numpy())
    return np.stack(out)


def served(port, texts=TEXTS, model="bge-m3"):
    body = json.dumps({"model": model, "input": texts}).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/embeddings",
        body,
        {"Content-Type": "application/json"},
    )
    data = json.load(urllib.request.urlopen(req, timeout=300))["data"]
    emb = np.array([d["embedding"] for d in sorted(data, key=lambda d: d["index"])])
    return emb / np.linalg.norm(emb, axis=1, keepdims=True)


def random_check(args):
    """Random-token texts at the bench lengths, server vs fp32 HF on --device."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = (
        AutoModel.from_pretrained(MODEL, torch_dtype=torch.float32)
        .to(args.device)
        .eval()
    )
    rng = np.random.default_rng(0)
    allowed = np.array(sorted(set(range(tok.vocab_size)) - set(tok.all_special_ids)))
    lens = [(1024, 4096, 8096)[i % 3] for i in range(args.random)]
    texts = [tok.decode(rng.choice(allowed, n).tolist()) for n in lens]
    cos, ntok = [], []
    for i in range(0, len(texts), 8):
        got = served(args.port, texts[i : i + 8], args.model)
        for t, g in zip(texts[i : i + 8], got):
            x = tok(t, return_tensors="pt", truncation=True, max_length=8192).to(
                args.device
            )
            with torch.no_grad():
                r = torch.nn.functional.normalize(
                    model(**x).last_hidden_state[0, 0], dim=-1
                )
            cos.append(float(np.dot(r.cpu().numpy(), g)))
            ntok.append(int(x["input_ids"].shape[1]))
    cos = np.array(cos)
    by_len = {
        str(n): round(float(cos[np.array(lens) == n].min()), 6)
        for n in (1024, 4096, 8096)
    }
    return {
        "n": len(cos),
        "min_cos": float(cos.min()),
        "mean_cos": float(cos.mean()),
        "min_cos_by_len": by_len,
        "tokens_min_max": [min(ntok), max(ntok)],
        "cos": [round(float(c), 6) for c in cos],
    }


def ranking(emb):
    q, p = emb[: len(QUERIES)], emb[len(QUERIES) : len(QUERIES) + len(PASSAGES)]
    return (q @ p.T).argmax(axis=1).tolist()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--port", type=int)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out")
    ap.add_argument("--model", default="bge-m3", help="served model name")
    ap.add_argument(
        "--random",
        type=int,
        default=0,
        help="instead: N random texts of 1024/4096/8096 tokens vs the fp32 reference",
    )
    args = ap.parse_args()
    if args.random:
        res = random_check(args)
        print(json.dumps({k: v for k, v in res.items() if k != "cos"}))
        if args.out:
            with open(args.out, "w") as f:
                json.dump(res, f, indent=1)
        return
    if args.port is None:
        np.savez(args.ref, emb=reference(args.device))
        print("saved", args.ref)
        return
    ref = np.load(args.ref)["emb"]
    got = served(args.port, model=args.model)
    cos = (ref * got).sum(axis=1)
    res = {
        "min_cos": float(cos.min()),
        "cos": [round(float(c), 6) for c in cos],
        "rank_ref": ranking(ref),
        "rank_served": ranking(got),
        "rank_match": ranking(ref) == ranking(got),
    }
    print(json.dumps(res))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
