# Copyright (c) 2026 BAAI. All rights reserved.
"""Dense check from vLLM's official bge-m3 test, against a running server.

Same sentences, reference matrix and tolerance as
tests/entrypoints/pooling/embed/test_bge_m3.py::test_bge_m3_api_server_embedding
(reference values from the BAAI/bge-m3 model card), but without RemoteOpenAIServer
so it can target an already started server.

  python3 check_official_bge_m3.py --port 8000 [--model BAAI/bge-m3] [--out res.json]
"""

import argparse
import json

import openai
import torch

sentences_1 = ["What is BGE M3?", "Definition of BM25"]
sentences_2 = [
    "BGE M3 is an embedding model supporting dense retrieval, "
    "lexical matching and multi-vector interaction.",
    "BM25 is a bag-of-words retrieval function that ranks a set "
    "of documents based on the query terms appearing in each document",
]
similarity_reference = [[0.6259, 0.3474], [0.3309, 0.6734]]


def run_client_embeddings(client, model, sentences):
    # Same as embed_utils.run_client_embeddings in the vLLM tests.
    outputs = client.embeddings.create(input=sentences, model=model)
    return [data.embedding for data in outputs.data]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--model", default="BAAI/bge-m3")
    ap.add_argument("--out")
    args = ap.parse_args()
    client = openai.OpenAI(base_url=f"http://127.0.0.1:{args.port}/v1", api_key="EMPTY")
    embeddings_1 = torch.tensor(run_client_embeddings(client, args.model, sentences_1))
    embeddings_2 = torch.tensor(run_client_embeddings(client, args.model, sentences_2))
    similarity = embeddings_1 @ embeddings_2.T
    reference = torch.tensor(similarity_reference)
    passed = bool(torch.allclose(similarity, reference, rtol=0.01))
    rel = ((similarity - reference).abs() / reference.abs()).max().item()
    res = {
        "port": args.port,
        "similarity": similarity.tolist(),
        "reference": similarity_reference,
        "max_rel_err": rel,
        "rtol": 0.01,
        "passed": passed,
    }
    print(json.dumps(res))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
