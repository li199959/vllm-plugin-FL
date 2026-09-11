import torch
import torch.nn.functional as F
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm_fl.dispatch import CachedOp

_silu_and_mul = CachedOp("silu_and_mul")
_gelu_and_mul = CachedOp("gelu_and_mul")
_swigluoai_uninterleave = CachedOp("swigluoai_uninterleave")
_swigluoai_interleaved = CachedOp("swigluoai_interleaved")


def apply_moe_activation(
    activation: MoEActivation,
    output: torch.Tensor,
    input: torch.Tensor,
    *,
    clamp_limit: float | None = None,
    alpha: float | None = None,
    beta: float | None = None,
) -> torch.Tensor:
    """Apply MoE activation function.

    ``clamp_limit``/``alpha``/``beta`` (from the layer's quant config) drive
    the clamped SwiGLU variants, mirroring upstream ``apply_moe_activation``.
    """
    assert input.dim() == 2, "Input must be 2D"
    assert output.dim() == 2, "Output must be 2D"
    if activation.is_gated:
        assert output.size(-1) * 2 == input.size(-1), (
            f"{activation.value} expects 2x ratio: "
            f"{output.size(-1) * 2} vs {input.size(-1)}"
        )
    else:
        assert output.size(-1) == input.size(-1), (
            f"{activation.value} expects equal sizes: "
            f"{output.size(-1)} vs {input.size(-1)}"
        )

    # Activations with gated multiplication (gate × activation(up))
    if activation == MoEActivation.SILU:
        output.copy_(_silu_and_mul(None, input))
    elif activation == MoEActivation.GELU:
        output.copy_(_gelu_and_mul(None, input))
    elif activation == MoEActivation.SWIGLUOAI:
        # gpt-oss layout: gate/up interleaved ([g0, u0, g1, u1, ...]).
        # Single fused triton kernel via the FL dispatch; the pure-torch
        # fallback decomposes into a mul/sigmoid/clamp micro-kernel storm.
        limit = 7.0 if clamp_limit is None else clamp_limit
        a = 1.702 if alpha is None else alpha
        b = 1.0 if beta is None else beta
        try:
            _swigluoai_interleaved(output, input, limit, a, b)
        except Exception:
            gate = input[..., ::2].float().clamp(max=limit)
            up = input[..., 1::2].float().clamp(min=-limit, max=limit)
            output.copy_(
                (gate * torch.sigmoid(gate * a) * (up + b)).to(output.dtype)
            )
    elif activation == MoEActivation.SWIGLUOAI_UNINTERLEAVE:
        # MiniMax-M3 layout: w13 packed as [all gates | all ups].
        # gate * sigmoid(alpha * gate) * (up + beta), both halves clamped;
        # computed in fp32 by the fused triton kernel, falling back to the
        # pure-torch decomposition only if dispatch has no implementation.
        assert clamp_limit is not None, (
            "SWIGLUOAI_UNINTERLEAVE requires clamp_limit"
        )
        a = 1.702 if alpha is None else alpha
        b = 1.0 if beta is None else beta
        try:
            _swigluoai_uninterleave(output, input, clamp_limit, a, b)
        except Exception:
            half = input.size(-1) // 2
            gate = input[..., :half].float().clamp(max=clamp_limit)
            up = input[..., half:].float().clamp(
                min=-clamp_limit, max=clamp_limit
            )
            output.copy_(
                (gate * torch.sigmoid(a * gate) * (up + b)).to(output.dtype)
            )
    elif activation == MoEActivation.SWIGLUSTEP:
        from vllm.model_executor.layers.activation import swiglustep_and_mul_triton

        swiglustep_and_mul_triton(output, input)

    # Activations without gated multiplication
    elif activation == MoEActivation.SILU_NO_MUL:
        output.copy_(F.silu(input))
    elif activation == MoEActivation.GELU_NO_MUL:
        output.copy_(F.gelu(input))
    elif activation == MoEActivation.RELU2_NO_MUL:
        F.relu(input, inplace=True)
        torch.square(input, out=output)
    else:
        raise ValueError(f"Unsupported FusedMoe activation: {activation}")

    return output
