# Copyright (c) 2026 BAAI. All rights reserved.
"""Pure-torch fallbacks for ``_moe_C::topk_sigmoid`` / ``_moe_C::topk_softmax``.

Mirrors ``csrc/libtorch_stable/moe/topk_softmax_kernels.cu`` from upstream
vLLM so sigmoid/softmax MoE routing works on builds without the native
``_moe_C`` extension:

- scores = softmax(gating) or sigmoid(gating), computed in fp32; NaN/Inf
  scores are clamped to 0 (CUDA-graph padding tokens produce NaN logits).
- ``bias`` (e_score_correction_bias) participates in expert *selection*
  only; the emitted weights are the unbiased scores.
- top-k selection; weights renormalized by the selected sum when
  ``renormalize`` (denominator clamped to 1 when the sum is not positive).
- ``token_expert_indices`` is a scratch buffer discarded by the python
  wrapper, matching the kernel we leave it untouched.
"""

import torch

_SIGMOID_SCHEMA = (
    "topk_sigmoid(Tensor! topk_weights, Tensor! topk_indices, Tensor! "
    "token_expert_indices, Tensor gating_output, bool renormalize, Tensor? "
    "bias) -> ()"
)
_SOFTMAX_SCHEMA = (
    "topk_softmax(Tensor! topk_weights, Tensor! topk_indices, Tensor! "
    "token_expert_indices, Tensor gating_output, bool renormalize, Tensor? "
    "bias) -> ()"
)


def _topk_routing(
    topk_weights: torch.Tensor,
    topk_indices: torch.Tensor,
    gating_output: torch.Tensor,
    renormalize: bool,
    bias: torch.Tensor | None,
    softmax: bool,
) -> None:
    scores = gating_output.float()
    if softmax:
        scores = torch.softmax(scores, dim=-1)
    else:
        scores = torch.sigmoid(scores)
    scores = torch.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)

    if bias is not None:
        scores_for_choice = scores + bias.float()
    else:
        scores_for_choice = scores

    # Iterative argmax with lowest-index tie-breaking, mirroring the CUDA
    # kernel: all-equal rows (e.g. NaN logits clamped to 0 in CUDA-graph
    # padding) must yield experts [0, 1, ..., k-1] like the kernel's index
    # tie-break, not torch.topk's arbitrary tie order.
    num_tokens, num_experts = scores.shape
    topk = topk_weights.shape[-1]
    selected = torch.zeros(
        num_tokens, num_experts, dtype=torch.bool, device=scores.device
    )
    index_list = []
    for _ in range(topk):
        masked = scores_for_choice.masked_fill(selected, float("-inf"))
        expert = torch.argmax(masked, dim=-1)
        index_list.append(expert)
        # scatter_ (not setitem): boolean advanced indexing goes through an
        # index_put_ path that is not capturable on this stack.
        selected.scatter_(1, expert.unsqueeze(1), True)
    indices = torch.stack(index_list, dim=1)

    weights = scores.gather(1, indices)

    if renormalize:
        denom = weights.sum(dim=-1, keepdim=True)
        weights = weights / torch.where(denom > 0, denom, torch.ones_like(denom))

    topk_weights.copy_(weights.to(topk_weights.dtype))
    topk_indices.copy_(indices.to(topk_indices.dtype))


def _topk_sigmoid(topk_weights, topk_indices, token_expert_indices,
                  gating_output, renormalize, bias):
    _topk_routing(topk_weights, topk_indices, gating_output, renormalize,
                  bias, softmax=False)


def _topk_softmax(topk_weights, topk_indices, token_expert_indices,
                  gating_output, renormalize, bias):
    _topk_routing(topk_weights, topk_indices, gating_output, renormalize,
                  bias, softmax=True)


def register_moe_topk_ops() -> None:
    """Define the _moe_C topk op schemas + CUDA fallbacks when needed."""
    if getattr(register_moe_topk_ops, "_done", False):
        return

    try:
        import vllm._moe_C  # noqa: F401

        register_moe_topk_ops._done = True
        return  # native build already provides the kernels
    except (ImportError, OSError):
        pass

    lib = torch.library.Library("_moe_C", "FRAGMENT")
    for schema, fn in (
        (_SIGMOID_SCHEMA, _topk_sigmoid),
        (_SOFTMAX_SCHEMA, _topk_softmax),
    ):
        lib.define(schema)
        lib.impl(schema.split("(")[0], fn, "CUDA")
    # Keep the Library alive: a garbage-collected FRAGMENT library
    # unregisters its schemas and kernels with it.
    register_moe_topk_ops._lib = lib
    register_moe_topk_ops._done = True
