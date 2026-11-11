# Copyright (c) 2026 BAAI. All rights reserved.
"""BEIR retrieval eval (MTEB protocol) for an OpenAI-compatible embedding server.

  python3 beir_eval.py --port 8200 --tasks scifact arguana --out res.json

Data: ModelScope ``mteb/<task>`` (corpus.jsonl, queries.jsonl, qrels/test.tsv).
Documents are "title text" joined with a space (MTEB's default for BEIR
corpora), queries are used as-is (bge-m3 needs no instruction), embeddings are
L2-normalized, retrieval is exact cosine top-1000, and the metric is nDCG@10
with graded relevance from qrels (pytrec_eval convention). For ArguAna the query
itself is excluded from its own result list, as MTEB does.
"""

import argparse
import json
import math
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np


def download(task):
    from modelscope import snapshot_download

    return Path(snapshot_download(f"mteb/{task}", repo_type="dataset"))


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def load_task(root):
    corpus = {
        d["_id"]: (d.get("title", "") + " " + d.get("text", "")).strip()
        for d in load_jsonl(root / "corpus.jsonl")
    }
    queries = {d["_id"]: d["text"] for d in load_jsonl(root / "queries.jsonl")}
    qrels = {}
    with open(root / "qrels" / "test.tsv") as f:
        for i, line in enumerate(f):
            parts = line.rstrip("\n").split("\t")
            if i == 0 and not parts[-1].lstrip("-").isdigit():
                continue  # header
            qid, did, score = parts[0], parts[1], int(parts[2])
            if score > 0:
                qrels.setdefault(qid, {})[did] = score
    queries = {q: t for q, t in queries.items() if q in qrels}
    return corpus, queries, qrels


def embed(port, model, texts, batch=64, workers=8):
    url = f"http://127.0.0.1:{port}/v1/embeddings"

    def call(chunk):
        body = json.dumps({"model": model, "input": chunk}).encode()
        req = urllib.request.Request(url, body, {"Content-Type": "application/json"})
        data = json.load(urllib.request.urlopen(req, timeout=600))["data"]
        return [d["embedding"] for d in sorted(data, key=lambda d: d["index"])]

    chunks = [texts[i : i + batch] for i in range(0, len(texts), batch)]
    with ThreadPoolExecutor(workers) as ex:
        out = [e for part in ex.map(call, chunks) for e in part]
    emb = np.asarray(out, dtype=np.float32)
    return emb / np.linalg.norm(emb, axis=1, keepdims=True)


def ndcg_at_k(ranked, rels, k=10):
    dcg = sum(
        (2 ** rels.get(d, 0) - 1) / math.log2(i + 2) for i, d in enumerate(ranked[:k])
    )
    ideal = sorted(rels.values(), reverse=True)[:k]
    idcg = sum((2**r - 1) / math.log2(i + 2) for i, r in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def evaluate(task, port, model):
    root = download(task)
    corpus, queries, qrels = load_task(root)
    doc_ids, q_ids = list(corpus), list(queries)
    t0 = time.time()
    d_emb = embed(port, model, [corpus[d] for d in doc_ids])
    q_emb = embed(port, model, [queries[q] for q in q_ids])
    enc_s = time.time() - t0
    scores = q_emb @ d_emb.T
    exclude_self = task == "arguana"
    pos = {d: i for i, d in enumerate(doc_ids)}
    ndcgs, recalls = [], []
    for qi, qid in enumerate(q_ids):
        s = scores[qi]
        if exclude_self and qid in pos:
            s = s.copy()
            s[pos[qid]] = -np.inf
        top = np.argpartition(-s, 100)[:100]
        ranked = [doc_ids[i] for i in top[np.argsort(-s[top])]]
        ndcgs.append(ndcg_at_k(ranked, qrels[qid]))
        rel = set(qrels[qid])
        recalls.append(len(rel & set(ranked[:100])) / len(rel))
    return {
        "task": task,
        "num_docs": len(doc_ids),
        "num_queries": len(q_ids),
        "ndcg@10": round(100 * float(np.mean(ndcgs)), 2),
        "recall@100": round(100 * float(np.mean(recalls)), 2),
        "encode_seconds": round(enc_s, 1),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--model", default="BAAI/bge-m3")
    ap.add_argument("--tasks", nargs="+", default=["scifact", "arguana"])
    ap.add_argument("--out")
    args = ap.parse_args()
    results = []
    for task in args.tasks:
        r = evaluate(task, args.port, args.model)
        print(json.dumps(r), flush=True)
        results.append(r)
    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
