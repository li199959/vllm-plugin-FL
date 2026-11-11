#!/usr/bin/env python3
"""Count broken generations in EvalScope prediction files: outputs containing a
run of >=8 '!' (token id 0 in the Qwen vocab, typical of NaN logits) and
non-'stop' finishes. Usage: count_garbage.py <results_dir> [...]"""

import glob
import json
import sys

for root in sys.argv[1:]:
    total = bang = cut = 0
    for f in glob.glob(f"{root}/predictions/**/*.jsonl", recursive=True):
        with open(f) as fh:
            for line in fh:
                c = json.loads(line)["model_output"]["choices"][0]
                m = c["message"]["content"]
                if isinstance(m, list):
                    m = " ".join(x.get("text", "") for x in m if isinstance(x, dict))
                total += 1
                bang += "!" * 8 in (m or "")
                cut += (c.get("stop_reason") or c.get("finish_reason")) != "stop"
    print(f"{root}: n={total} bang_runs={bang} non_stop={cut}")
