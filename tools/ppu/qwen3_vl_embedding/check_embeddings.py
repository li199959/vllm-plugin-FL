"""Compare a running vLLM Qwen3-VL-Embedding server against a HF reference.

Inputs follow the model's own scripts/qwen3_vl_embedding.py: a system message
with the instruction, the user content (text / image / both), chat template with
add_generation_prompt=True, last-token hidden state, L2 normalize. The server is
called through /v1/embeddings with the same messages and add_generation_prompt.

Usage: [CUDA_VISIBLE_DEVICES=<free gpu>] python check_embeddings.py <port> [--save ref.pt] [--load ref.pt]
"""

import argparse
import base64
import io
import json
import urllib.request

import torch
import torch.nn.functional as F
from PIL import Image

MODEL = "/mnt/cpfs/xiangbin/Qwen3-VL-Embedding-8B"
IMAGES = {
    "logo": "/mnt/cpfs/xiangbin/codex/Unlimited-OCR/assets/baidu.png",
    "doc": "/mnt/cpfs/xiangbin/codex/profiles/uocr_samples/doc_small.png",
    "frame": "/mnt/cpfs/xiangbin/codex/profiles/uocr_samples/gif_frame_140.png",
}
DEFAULT_INSTRUCTION = "Represent the user's input."
QUERY_INSTRUCTION = "Retrieve images or text relevant to the user's query."

# (name, instruction, text, image key)
SAMPLES = [
    ("q_en", QUERY_INSTRUCTION, "What is the capital of France?", None),
    ("q_zh", QUERY_INSTRUCTION, "如何训练一个大语言模型？", None),
    ("q_logo", QUERY_INSTRUCTION, "The logo of a Chinese search engine company.", None),
    ("q_doc", QUERY_INSTRUCTION, "A scanned page of a printed document.", None),
    (
        "d_en",
        DEFAULT_INSTRUCTION,
        "Paris is the capital and most populous city of France.",
        None,
    ),
    (
        "d_zh",
        DEFAULT_INSTRUCTION,
        "训练大语言模型需要大规模语料、分布式训练框架以及对齐阶段的人类反馈。",
        None,
    ),
    (
        "d_de",
        DEFAULT_INSTRUCTION,
        "Berlin ist die Hauptstadt der Bundesrepublik Deutschland.",
        None,
    ),
    (
        "d_ja",
        DEFAULT_INSTRUCTION,
        "東京は日本の首都であり、世界有数の大都市です。",
        None,
    ),
    (
        "d_long",
        DEFAULT_INSTRUCTION,
        " ".join(
            [
                "Large language models are trained on diverse corpora and evaluated on many benchmarks."
            ]
            * 120
        ),
        None,
    ),
    ("i_logo", DEFAULT_INSTRUCTION, None, "logo"),
    ("i_doc", DEFAULT_INSTRUCTION, None, "doc"),
    ("i_frame", DEFAULT_INSTRUCTION, None, "frame"),
    ("ti_doc", DEFAULT_INSTRUCTION, "A document page with printed text.", "doc"),
]
# query -> expected best document among the d_*/i_* entries
EXPECT = {
    "q_en": ("d_en",),
    "q_zh": ("d_zh",),
    "q_logo": ("i_logo",),
    "q_doc": ("i_doc", "ti_doc"),
}  # ti_doc holds the same page plus a caption


def conversation(instruction, text, image, as_url):
    content = []
    if image:
        if as_url:
            buf = io.BytesIO()
            Image.open(IMAGES[image]).convert("RGB").save(buf, format="PNG")
            url = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
            content.append({"type": "image_url", "image_url": {"url": url}})
        else:
            content.append({"type": "image"})
    if text:
        content.append({"type": "text", "text": text})
    return [
        {"role": "system", "content": [{"type": "text", "text": instruction}]},
        {"role": "user", "content": content},
    ]


@torch.no_grad()
def hf_reference():
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    processor = AutoProcessor.from_pretrained(MODEL)
    model = (
        Qwen3VLForConditionalGeneration.from_pretrained(
            MODEL, dtype=torch.bfloat16, attn_implementation="sdpa"
        )
        .to("cuda")
        .eval()
    )
    out = {}
    for name, instr, text, image in SAMPLES:
        conv = conversation(instr, text, image, as_url=False)
        prompt = processor.apply_chat_template(
            conv, add_generation_prompt=True, tokenize=False
        )
        images = [Image.open(IMAGES[image]).convert("RGB")] if image else None
        inputs = processor(text=[prompt], images=images, return_tensors="pt").to("cuda")
        hidden = model.model(**inputs).last_hidden_state
        out[name] = (
            F.normalize(hidden[0, -1].float(), dim=-1).cpu(),
            int(inputs["input_ids"].shape[1]),
        )
    return out


def server_embeddings(port):
    out = {}
    for name, instr, text, image in SAMPLES:
        # The tokenizer post-processor appends <|endoftext|>, and the official
        # pipeline pools that token. vLLM chat requests default to
        # add_special_tokens=false, so request it explicitly.
        body = {
            "model": "qwen3-vl-embedding",
            "messages": conversation(instr, text, image, as_url=True),
            "add_generation_prompt": True,
            "add_special_tokens": True,
            "encoding_format": "float",
        }
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/v1/embeddings",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        d = json.load(urllib.request.urlopen(req, timeout=600))
        out[name] = (
            torch.tensor(d["data"][0]["embedding"]),
            d["usage"]["prompt_tokens"],
        )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("port", type=int)
    ap.add_argument("--save")
    ap.add_argument("--load")
    a = ap.parse_args()
    ref = torch.load(a.load) if a.load else hf_reference()
    if a.save:
        torch.save(ref, a.save)
    got = server_embeddings(a.port)
    worst = 1.0
    for name, *_ in SAMPLES:
        r, rn = ref[name]
        g, gn = got[name]
        cos = float(F.cosine_similarity(r, g, dim=0))
        worst = min(worst, cos)
        print(f"{name:8s} tokens hf={rn:5d} vllm={gn:5d}  cos={cos:.6f}")
    docs = [n for n, *_ in SAMPLES if n[0] in "di" or n.startswith("ti")]
    ok = 0
    for q, want in EXPECT.items():
        for label, emb in (("hf", ref), ("vllm", got)):
            scores = {d: float(emb[q][0] @ emb[d][0]) for d in docs}
            best = max(scores, key=scores.get)
            ok += best in want
            print(
                f"retrieval {label:4s} {q:7s} -> {best:7s} (want {'/'.join(want)}) "
                f"{'OK' if best in want else 'MISS'}"
            )
    print(f"min cosine {worst:.6f}; retrieval {ok}/{2 * len(EXPECT)} OK")


if __name__ == "__main__":
    main()
