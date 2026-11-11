"""Compare /rerank scores of two Qwen3-Reranker servers (reference vs candidate).

Each query has one relevant and several distractor documents (zh/en, short and
~2k-token long docs). Reports per-pair score difference and whether both servers
rank the relevant document first.

Usage: python check_scores.py <ref_port> <cand_port>
"""

import json
import sys
import urllib.request

LONG = " ".join(
    [
        "The quarterly report covers revenue, operating costs, staffing and regional growth."
    ]
    * 160
)
CASES = [
    (
        "What is the capital of France?",
        [
            "Paris is the capital and most populous city of France.",
            "Berlin is the capital of Germany.",
            "The Eiffel Tower was completed in 1889.",
            "France is famous for its wine and cheese.",
        ],
    ),
    (
        "如何训练一个大语言模型？",
        [
            "训练大语言模型需要大规模语料、分布式训练框架以及对齐阶段的人类反馈。",
            "北京是中国的首都。",
            "大熊猫主要生活在四川的山区。",
            "这家餐厅的火锅很好吃。",
        ],
    ),
    (
        "Explain how photosynthesis works.",
        [
            "Photosynthesis converts light energy, water and carbon dioxide into glucose and oxygen in chloroplasts.",
            "The stock market closed higher today.",
            "Mitochondria produce ATP through cellular respiration.",
            LONG,
        ],
    ),
    (
        "What did the quarterly report say about staffing?",
        [
            LONG + " Staffing grew by 12 percent, mostly in engineering.",
            "Photosynthesis happens in chloroplasts.",
            "The weather will be sunny tomorrow.",
            "Quarterly reports are published four times a year.",
        ],
    ),
]


def rerank(port, query, docs):
    body = {"model": "Qwen3-Reranker-8B", "query": query, "documents": docs}
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/rerank",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    d = json.load(urllib.request.urlopen(req, timeout=600))
    scores = [0.0] * len(docs)
    for r in d["results"]:
        scores[r["index"]] = r["relevance_score"]
    return scores


def main():
    ref_port, cand_port = int(sys.argv[1]), int(sys.argv[2])
    worst, ok = 0.0, 0
    for q, docs in CASES:
        a, b = rerank(ref_port, q, docs), rerank(cand_port, q, docs)
        diffs = [abs(x - y) for x, y in zip(a, b)]
        worst = max(worst, *diffs)
        top_a, top_b = (
            max(range(len(a)), key=a.__getitem__),
            max(range(len(b)), key=b.__getitem__),
        )
        ok += (top_a == 0) + (top_b == 0)
        print(f"{q[:40]:40s} ref {[round(x, 4) for x in a]}")
        print(
            f"{'':40s} new {[round(x, 4) for x in b]}  max|diff| {max(diffs):.4f}  top ref={top_a} new={top_b}"
        )
    print(
        f"max |score diff| {worst:.4f}; relevant doc ranked first {ok}/{2 * len(CASES)}"
    )


if __name__ == "__main__":
    main()
