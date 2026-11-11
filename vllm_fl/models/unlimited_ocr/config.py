# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Ported from vllm-project/vllm#46564 / #47102 (Unlimited-OCR, vLLM >= 0.25) to v0.24.
"""Configs for Unlimited-OCR (baidu/Unlimited-OCR).

Unlimited-OCR reuses the DeepSeek-OCR multimodal layout (DeepEncoder = SAM-ViT-B +
CLIP-L, a linear MLP projector and a DeepSeek-V2 text backbone). The language model
is a DeepSeek-V2 *MoE* with plain multi-head attention (``use_mla=False``), so the
nested config is parsed with ``DeepseekVLV2Config``.
"""

from typing import TYPE_CHECKING

from vllm.logger import init_logger
from vllm.model_executor.models.config import VerifyAndUpdateConfig
from vllm.transformers_utils.configs.deepseek_vl2 import DeepseekVLV2Config

if TYPE_CHECKING:
    from vllm.config import ModelConfig, VllmConfig

logger = init_logger(__name__)

# Generated tokens attend the prompt plus this many most recent tokens.
DEFAULT_RSWA_WINDOW = 128


class UnlimitedOCRConfig(DeepseekVLV2Config):
    model_type = "unlimited_ocr"

    # Transformers v5 builds nested sub-configs from each concrete class'
    # ``__init__`` signature, so an explicit ``__init__`` is required.
    def __init__(
        self,
        tile_tag: str = "2D",
        global_view_pos: str = "head",
        candidate_resolutions: tuple[tuple[int, int]] = ((384, 384),),
        rswa_window: int | None = None,
        **kwargs,
    ):
        language_config = kwargs.get("language_config") or {}
        # The checkpoint stores the window as language_config.sliding_window_size.
        if rswa_window is None:
            rswa_window = language_config.get(
                "sliding_window_size", DEFAULT_RSWA_WINDOW
            )
        super().__init__(
            tile_tag=tile_tag,
            global_view_pos=global_view_pos,
            candidate_resolutions=candidate_resolutions,
            **kwargs,
        )
        self.rswa_window = rswa_window
        # Plain MHA backbone (qk_nope_head_dim == qk_rope_head_dim == 0): under
        # Transformers v5 DeepseekV2Config reports head_dim = qk_nope + qk_rope = 0,
        # which makes ModelConfig.get_head_size() return 0. The Triton backend then
        # sizes its split-softmax buffers for head size 1 and the 3D decode kernel
        # writes past them. The real head size is hidden_size / num_heads (128).
        text = self.text_config
        if not getattr(text, "head_dim", None):
            text.head_dim = text.hidden_size // text.num_attention_heads


class UnlimitedOCRForCausalLMConfig(VerifyAndUpdateConfig):
    @staticmethod
    def verify_and_update_config(vllm_config: "VllmConfig") -> None:
        """R-SWA is implemented in the Triton unified-attention path only."""
        from vllm.v1.attention.backends.registry import AttentionBackendEnum

        attn_config = vllm_config.attention_config
        if attn_config.backend is None:
            attn_config.backend = AttentionBackendEnum.TRITON_ATTN
        elif attn_config.backend != AttentionBackendEnum.TRITON_ATTN:
            raise ValueError(
                "Unlimited-OCR: R-SWA is only implemented for the TRITON_ATTN backend, "
                f"got {attn_config.backend!r}."
            )
        # R-SWA decode KV is not a causal function of the prefix, so it cannot be
        # reused across requests; single-turn OCR prompts rarely share a prefix.
        cache_config = vllm_config.cache_config
        if cache_config.enable_prefix_caching:
            cache_config.enable_prefix_caching = False
            logger.info("Unlimited-OCR: disabling prefix caching (R-SWA).")

    @staticmethod
    def verify_and_update_model_config(model_config: "ModelConfig") -> None:
        hf_config = model_config.hf_config
        hf_config.text_config.architectures = ["DeepseekV2ForCausalLM"]
        if getattr(hf_config, "rswa_window", None) is None:
            hf_config.rswa_window = DEFAULT_RSWA_WINDOW
