"""Compare complete M3 reports without changing their coverage accounting."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--baseline-bench", default="bench")
    parser.add_argument("--candidate-bench", default="bench")
    args = parser.parse_args()
    runs = []
    for root, bench_name in [
        (args.baseline, args.baseline_bench),
        (args.candidate, args.candidate_bench),
    ]:
        coverage = json.loads((root / "gpu_coverage/summary.json").read_text())
        verification = json.loads((root / "gpu_coverage/verification.json").read_text())
        assert verification["checks_passed"]
        assert coverage["capture"]["worker_count"] == 16
        assert all(coverage["capture"]["graph_replay_kernel_events"])
        bench = {}
        for round_name in ("warmup1", "warmup2", "profile"):
            data = json.loads((root / bench_name / (round_name + ".json")).read_text())
            assert data["completed"] == 128 and data["failed"] == 0
            assert (
                data["total_input_tokens"] == 524288
                and data["total_output_tokens"] == 131072
            )
            bench[round_name] = {
                k: data[k]
                for k in (
                    "duration",
                    "output_throughput",
                    "mean_ttft_ms",
                    "mean_tpot_ms",
                )
            }
        command = root / (
            "bench.valid.json" if bench_name == "bench_valid" else "bench.json"
        )
        runs.append(
            {
                "path": str(root.resolve()),
                "coverage": {k: coverage[k] for k in ("aten", "kernel", "combined")},
                "bench": bench,
                "command": json.loads(command.read_text()),
            }
        )
    assert runs[0]["command"] == runs[1]["command"], "Benchmark cases differ"
    baseline, candidate = runs
    delta = {
        k: (candidate["bench"]["profile"][k] / baseline["bench"]["profile"][k] - 1)
        * 100
        for k in baseline["bench"]["profile"]
    }
    meets = (
        candidate["coverage"]["combined"]["percent"] >= 80
        and delta["output_throughput"] >= -5
    )
    result = {
        "baseline": baseline,
        "candidate": candidate,
        "profile_change_percent": delta,
        "meets_coverage_and_throughput": meets,
        "latency_increases_over_five_percent": [
            k for k in ("mean_ttft_ms", "mean_tpot_ms") if delta[k] > 5
        ],
        "notes": [
            "All warmup results retained; acceptance comparison uses the identically profiled third round.",
            "Coverage is operator-type coverage, not GPU-time coverage.",
            "A single measured comparison is not a statistical confidence interval.",
        ],
    }
    (args.candidate / "performance_comparison.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    lines = [
        "MiniMax-M3：覆盖率与性能对照",
        "",
        "目标：合计覆盖率 ≥80%，相同 case 吞吐下降 ≤5%。",
        "",
        "| 指标 | 原配置复测 | 当前候选 |",
        "|---|---:|---:|",
    ]
    for key, label in [
        ("aten", "GPU 关联 ATen"),
        ("kernel", "独立 GPU kernel"),
        ("combined", "合计覆盖率"),
    ]:
        values = [run["coverage"][key] for run in runs]
        lines.append(
            "| "
            + label
            + " | "
            + " | ".join(
                f"{v['numerator']}/{v['denominator']}（{v['percent']:.2f}%）"
                for v in values
            )
            + " |"
        )
    for key, label in [
        ("output_throughput", "输出吞吐 tok/s"),
        ("mean_ttft_ms", "平均 TTFT ms"),
        ("mean_tpot_ms", "平均 TPOT ms"),
    ]:
        lines.append(
            f"| {label} | {baseline['bench']['profile'][key]:.2f} | {candidate['bench']['profile'][key]:.2f}（{delta[key]:+.2f}%） |"
        )
    lines += [
        "",
        f"覆盖率和吞吐双门槛：{'通过' if meets else '未通过'}。",
        "",
        "所有 16 个 worker trace 完整解析，有真实 graph replay，不变量核验通过。每轮 128 成功、0 失败，输入 524288、输出 131072 tokens。按精确 kernel 符号统计，没有人工分组合并。",
        "",
        "预热轮的数据保存在 performance_comparison.json。本表使用同样开启 profiler 的第三轮，不能把冷启动/编译耗时与稳定吞吐混比。单次对照不等于置信区间；TTFT 与 TPOT 单列，不能用吞吐覆盖延迟变化。",
        "",
    ]
    (args.candidate / "PERFORMANCE.md").write_text("\n".join(lines))
    print(
        json.dumps(
            {
                "coverage": candidate["coverage"]["combined"],
                "profile_change_percent": delta,
                "meets_coverage_and_throughput": meets,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
