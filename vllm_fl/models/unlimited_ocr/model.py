# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Ported from vllm-project/vllm#46564 / #47102 (Unlimited-OCR, vLLM >= 0.25) to v0.24.
"""Inference-only Unlimited-OCR model compatible with HuggingFace weights.

Unlimited-OCR (``baidu/Unlimited-OCR``) shares the DeepSeek-OCR (gundam,
``base_size=1024`` / ``image_size=640`` / crop) vision stack: a DeepEncoder
(SAM-ViT-B + CLIP-L) followed by a linear MLP projector, with the same image-token
tiling layout. The language backbone is a DeepSeek-V2 *MoE* (64 routed + 2 shared
experts, ``first_k_dense_replace=1``) with plain multi-head attention
(``qk_nope_head_dim == qk_rope_head_dim == 0``), which vLLM's
``DeepseekV2DecoderLayer`` already dispatches to ``DeepseekAttention``.

Differences from DeepSeek-OCR:

- up to 32 local crops (``dynamic_preprocess(max_num=32)``) instead of 6;
- multi-image requests fall back to non-crop mode, so the per-image processor
  output depends on the image count: multi-image requests bypass the processing
  cache (as ``DeepseekVL2MultiModalProcessor`` does) and their hashes include the
  crop flag;
- decode attention uses R-SWA (see ``rswa.py``).

Use ``NGramPerReqLogitsProcessor`` with
``extra_args={"ngram_size": 35, "window_size": 128}`` (1024 for multi-page input).
"""

import contextvars
import math
from collections.abc import Mapping, Sequence

from vllm.config import VllmConfig
from vllm.model_executor.models.deepseek_ocr import (
    DeepseekOCRDummyInputsBuilder,
    DeepseekOCRForCausalLM,
    DeepseekOCRMultiModalProcessor,
    DeepseekOCRProcessingInfo,
    NGramPerReqLogitsProcessor,
)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.hasher import MultiModalHasher
from vllm.multimodal.inputs import MultiModalKwargsItems
from vllm.multimodal.parse import (
    ImageEmbeddingItems,
    ImageProcessorItems,
    ImageSize,
    MultiModalDataItems,
)
from vllm.multimodal.processing import PromptReplacement, PromptUpdate
from vllm.multimodal.processing.context import TimingContext
from vllm.multimodal.processing.inputs import ProcessorInputs
from vllm.transformers_utils.processors import deepseek_ocr as _ocr_processing
from vllm.transformers_utils.processors.deepseek_ocr import (
    BASE_SIZE,
    CROP_MODE,
    IMAGE_SIZE,
    DeepseekOCRProcessor,
    count_tiles,
)

from . import rswa
from .config import UnlimitedOCRConfig

__all__ = [
    "NGramPerReqLogitsProcessor",
    "UnlimitedOCRForCausalLM",
]

# Unlimited-OCR supports up to 32 local crops (vs 6 for DeepSeek-OCR).
_UNLIMITED_OCR_MAX_CROPS = 32

# v0.24's DeepseekOCRProcessor calls the module-level dynamic_preprocess with the
# default max_num; route its max_num through a context variable instead.
_max_crops: contextvars.ContextVar[int | None] = contextvars.ContextVar(
    "unlimited_ocr_max_crops", default=None
)
_dynamic_preprocess = _ocr_processing.dynamic_preprocess


def _dynamic_preprocess_max_crops(image, *args, **kwargs):
    max_crops = _max_crops.get()
    if max_crops is not None and not args:
        kwargs.setdefault("max_num", max_crops)
    return _dynamic_preprocess(image, *args, **kwargs)


if (
    getattr(_ocr_processing.dynamic_preprocess, "__name__", "")
    != "_dynamic_preprocess_max_crops"
):
    _ocr_processing.dynamic_preprocess = _dynamic_preprocess_max_crops


class UnlimitedOCRProcessor(DeepseekOCRProcessor):
    """DeepseekOCRProcessor with 32 crops; multi-image input disables crop mode."""

    def __init__(self, tokenizer, max_crops: int = _UNLIMITED_OCR_MAX_CROPS, **kwargs):
        self.max_crops = max_crops
        super().__init__(tokenizer, **kwargs)

    def tokenize_with_images(
        self,
        conversation: str,
        images,
        bos: bool = True,
        eos: bool = True,
        cropping: bool = True,
    ):
        if len(images) > 1:
            cropping = False
        token = _max_crops.set(self.max_crops)
        try:
            return super().tokenize_with_images(
                conversation, images, bos=bos, eos=eos, cropping=cropping
            )
        finally:
            _max_crops.reset(token)


class UnlimitedOCRProcessingInfo(DeepseekOCRProcessingInfo):
    def get_hf_config(self):
        return self.ctx.get_hf_config(UnlimitedOCRConfig)

    def get_hf_processor(self, **kwargs: object):
        v1_processor_config = dict(
            image_size=IMAGE_SIZE,
            base_size=BASE_SIZE,
            crop_mode=CROP_MODE,
            strategy="v1",
            max_crops=_UNLIMITED_OCR_MAX_CROPS,
        )
        return self.ctx.get_hf_processor(
            UnlimitedOCRProcessor,
            **{**v1_processor_config, **kwargs},
        )

    def get_num_image_tokens(
        self, *, image_width: int, image_height: int, cropping: bool = True
    ) -> int:
        patch_size = 16
        downsample_ratio = 4

        if cropping:
            if image_width <= IMAGE_SIZE and image_height <= IMAGE_SIZE:
                crop_ratio = [1, 1]
            else:
                crop_ratio = count_tiles(
                    image_width,
                    image_height,
                    max_num=_UNLIMITED_OCR_MAX_CROPS,
                    image_size=IMAGE_SIZE,
                )
            num_width_tiles, num_height_tiles = crop_ratio
        else:
            num_width_tiles = num_height_tiles = 1

        h = w = math.ceil((BASE_SIZE // patch_size) / downsample_ratio)
        h2 = w2 = math.ceil((IMAGE_SIZE // patch_size) / downsample_ratio)

        global_views_tokens = h * (w + 1)
        if num_width_tiles > 1 or num_height_tiles > 1:
            local_views_tokens = (num_height_tiles * h2) * (num_width_tiles * w2 + 1)
        else:
            local_views_tokens = 0

        return global_views_tokens + local_views_tokens + 1

    def get_image_size_with_most_features(self) -> ImageSize:
        # A vertical grid maximizes both crop count and newline tokens.
        return ImageSize(width=IMAGE_SIZE, height=IMAGE_SIZE * _UNLIMITED_OCR_MAX_CROPS)


class UnlimitedOCRMultiModalProcessor(DeepseekOCRMultiModalProcessor):
    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        hf_processor = self.info.get_hf_processor(**hf_processor_mm_kwargs)
        crop_mode = hf_processor_mm_kwargs.get("crop_mode", CROP_MODE)

        image_token_id = hf_processor.image_token_id
        assert isinstance(image_token_id, int)

        def get_replacement_unlimited_ocr(item_idx: int):
            images = mm_items.get_items(
                "image", (ImageEmbeddingItems, ImageProcessorItems)
            )
            if isinstance(images, ImageEmbeddingItems):
                num_image_tokens = images.get_feature_size(item_idx)
            else:
                size = images.get_image_size(item_idx)
                # Must agree with UnlimitedOCRProcessor.tokenize_with_images.
                effective_cropping = bool(crop_mode) and len(images) == 1
                num_image_tokens = self.info.get_num_image_tokens(
                    image_width=size.width,
                    image_height=size.height,
                    cropping=effective_cropping,
                )
            return [image_token_id] * num_image_tokens

        return [
            PromptReplacement(
                modality="image",
                target=[image_token_id],
                replacement=get_replacement_unlimited_ocr,
            )
        ]

    def _apply_hf_processor(self, inputs: ProcessorInputs, timing_ctx: TimingContext):
        prompt_ids, mm_info, is_update_applied = super()._apply_hf_processor(
            inputs, timing_ctx
        )
        # Multi-image requests are processed without crops; keep their encoder
        # cache entries apart from the single-image (cropped) ones.
        images = inputs.mm_data_items.get("image")
        if isinstance(images, ImageProcessorItems) and len(images) > 1:
            hashes = dict(mm_info.hashes)
            hashes["image"] = [
                MultiModalHasher.hash_kwargs(mm_hash=h, cropping=False)
                for h in hashes["image"]
            ]
            mm_info = mm_info._replace(hashes=hashes)
        return prompt_ids, mm_info, is_update_applied

    def _cached_apply_hf_processor(
        self, inputs: ProcessorInputs, timing_ctx: TimingContext
    ):
        # Per-item output depends on how many images are in the request, which
        # the processing cache does not account for: cache single-image only.
        if inputs.mm_data_items.get_count("image", strict=False) > 1:
            return self._apply_hf_processor(inputs, timing_ctx)
        return super()._cached_apply_hf_processor(inputs, timing_ctx)


@MULTIMODAL_REGISTRY.register_processor(
    UnlimitedOCRMultiModalProcessor,
    info=UnlimitedOCRProcessingInfo,
    dummy_inputs=DeepseekOCRDummyInputsBuilder,
)
class UnlimitedOCRForCausalLM(DeepseekOCRForCausalLM):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        import torch

        super().__init__(vllm_config=vllm_config, prefix=prefix)
        rswa.install(
            window=int(vllm_config.model_config.hf_config.rswa_window),
            max_num_seqs=vllm_config.scheduler_config.max_num_seqs,
            device=torch.device("cuda", torch.cuda.current_device()),
        )
