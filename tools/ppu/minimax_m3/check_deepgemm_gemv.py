"""Which DeepGEMM nopad path is wrong at small M: gemm1 (K=6144) or gemm2 (K=192)?

Calls deep_gemm.m_grouped_gemm_bf16_bf16_bf16_nt_nopad on M3's two expert
GEMM shapes with the plugin's compact permute layout and compares each row to
an fp32 reference, twice per input (the failure is run-to-run varying). Also
times the call. ``--force-grouped`` bypasses the GEMV auto-selection.
"""

import argparse
import json

import torch
from sweep_small_m import measure

E, H, I, TOPK = 128, 6144, 192, 4


def grouped_call(force):
    import deep_gemm
    from deep_gemm.jit_kernels import m_grouped_gemm as mg

    if not force:
        return deep_gemm.m_grouped_gemm_bf16_bf16_bf16_nt_nopad
    original = mg.get_gemv_best_configs

    def no_gemv(*a, **kw):
        cfg = list(original(*a, **kw))
        cfg[1] = -1  # ThreadPerN == -1 makes the caller fall through to grouped GEMM
        return tuple(cfg)

    def call(lhs, rhs, out, m_indices, m_rows=None):
        mg.get_gemv_best_configs = no_gemv
        try:
            return deep_gemm.m_grouped_gemm_bf16_bf16_bf16_nt_nopad(
                lhs, rhs, out, m_indices, m_rows
            )
        finally:
            mg.get_gemv_best_configs = original

    return call


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rows",
        type=int,
        nargs="+",
        default=[4, 8, 16, 20, 21, 28, 32, 40, 47, 48, 49, 64],
    )
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--force-grouped", action="store_true")
    args = parser.parse_args()
    from deep_gemm.deep_gemm_tuner.deepgemm_tools import deepgemm_moe_permute

    call = grouped_call(args.force_grouped)
    torch.manual_seed(0)
    w1 = torch.randn(E, 2 * I, H, device="cuda", dtype=torch.bfloat16) * 0.05
    w2 = torch.randn(E, H, I, device="cuda", dtype=torch.bfloat16) * 0.02
    for m in args.rows:
        for gemm, (w, k) in {"gemm1": (w1, H), "gemm2": (w2, I)}.items():
            n = w.shape[1]
            bad_runs, max_err = 0, 0.0
            for seed in range(args.seeds):
                torch.manual_seed(1000 * m + seed)
                ids = torch.topk(
                    torch.randn(m, E, device="cuda"), TOPK, dim=-1
                ).indices.to(torch.int32)
                x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
                a, _, m_idx, _, m_rows = deepgemm_moe_permute(
                    x, None, ids, E, block_align=1, block_k=k
                )
                ref = torch.einsum("mk,mnk->mn", a.float(), w[m_idx.long()].float())
                for _ in range(2):
                    out = torch.full(
                        (a.shape[0], n),
                        float("nan"),
                        device="cuda",
                        dtype=torch.bfloat16,
                    )
                    call(a, w, out, m_idx, m_rows)
                    torch.cuda.synchronize()
                    err = (out.float() - ref).abs()
                    bad = (err > 0.02 + 0.02 * ref.abs()) | out.isnan()
                    bad_runs += int(bad.any())
                    max_err = max(max_err, float(err.nan_to_num(1e9).max()))
            a_t = torch.randn(m * TOPK, k, device="cuda", dtype=torch.bfloat16)
            ids = torch.topk(torch.randn(m, E, device="cuda"), TOPK, dim=-1).indices.to(
                torch.int32
            )
            a_t, _, m_idx, _, m_rows = deepgemm_moe_permute(
                torch.randn(m, k, device="cuda", dtype=torch.bfloat16),
                None,
                ids,
                E,
                block_align=1,
                block_k=k,
            )
            out = torch.empty((a_t.shape[0], n), device="cuda", dtype=torch.bfloat16)
            us = measure(lambda i: call(a_t, w, out, m_idx, m_rows), 1)
            gemv = k % 16 == 0 and (
                (m * TOPK <= 2 * E * 0.75 and k <= 256)
                or (m * TOPK < 0.65 * E and k > 256)
            )
            path = (
                "grouped(forced)"
                if args.force_grouped
                else ("gemv" if gemv else "grouped")
            )
            print(
                json.dumps(
                    {
                        "m": m,
                        "gemm": gemm,
                        "path": path,
                        "bad_runs": bad_runs,
                        "of": 2 * args.seeds,
                        "max_abs_err": round(max_err, 4),
                        "us": round(us, 1),
                    }
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
