import itertools
from collections.abc import Iterable, Set

import torch
from torch import nn

from vllm.config import PoolerConfig, VllmConfig, get_current_vllm_config
from vllm.model_executor.layers.pooler import (
    DispatchPooler,
    Pooler,
    PoolingParamsUpdate,
)
from vllm.model_executor.layers.pooler.activations import PoolerNormalize
from vllm.model_executor.layers.pooler.seqwise import (
    EmbeddingPoolerHead,
    pooler_for_embed,
)
from vllm.model_executor.layers.pooler.tokwise import (
    AllPool,
    StepPool,
    TokenEmbeddingPoolerHead,
    TokenPooler,
    pooler_for_token_classify,
    pooler_for_token_embed,
)
from vllm.model_executor.model_loader import DefaultModelLoader
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.adapters import _load_st_projector
from vllm.model_executor.models.roberta import RobertaEmbeddingModel
from vllm.pooling_params import PoolingParams
from vllm.tasks import PoolingTask
from vllm.v1.outputs import PoolerOutput
from vllm.v1.pool.metadata import PoolingMetadata


def filter_secondary_weights(
    all_weights: Iterable[tuple[str, torch.Tensor]],
    secondary_weight_prefixes: list[str],
) -> tuple[Iterable[tuple[str, torch.Tensor]], Iterable[tuple[str, torch.Tensor]]]:
    all_weights_1, all_weights_2 = itertools.tee(all_weights)

    def is_secondary(name: str) -> bool:
        return any(name.startswith(prefix) for prefix in secondary_weight_prefixes)

    secondary = (
        (name, weight) for name, weight in all_weights_1 if is_secondary(name)
    )
    primary = (
        (name, weight) for name, weight in all_weights_2 if not is_secondary(name)
    )
    return secondary, primary


class BgeM3EmbeddingModel(RobertaEmbeddingModel):
    """Backport the vLLM BGE-M3 embedding adapter to the FL compatibility layer."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        self.hidden_size = vllm_config.model_config.hf_config.hidden_size

        model_config = vllm_config.model_config
        self.head_dtype = model_config.head_dtype
        self.bos_token_id = model_config.hf_config.bos_token_id
        self.eos_token_id = model_config.hf_config.eos_token_id

        super().__init__(vllm_config=vllm_config, prefix=prefix)

        self.secondary_weight_prefixes = ["sparse_linear.", "colbert_linear."]
        self.secondary_weight_files = [
            weight_prefix + "pt" for weight_prefix in self.secondary_weight_prefixes
        ]
        self.secondary_weights = [
            DefaultModelLoader.Source(
                model_or_path=vllm_config.model_config.model,
                revision=None,
                prefix=weight_prefix,
                allow_patterns_overrides=[filename],
            )
            for filename, weight_prefix in zip(
                self.secondary_weight_files,
                self.secondary_weight_prefixes,
            )
        ]

    def _build_pooler(self, pooler_config: PoolerConfig) -> Pooler:
        self.sparse_linear = nn.Linear(self.hidden_size, 1, dtype=self.head_dtype)
        self.colbert_linear = nn.Linear(
            self.hidden_size,
            self.hidden_size,
            dtype=self.head_dtype,
        )

        vllm_config = get_current_vllm_config()
        model_config = vllm_config.model_config

        # Build the token_embed head with colbert_linear projector
        token_embed_head = TokenEmbeddingPoolerHead(
            head_dtype=self.head_dtype,
            projector=self.colbert_linear,
            activation=PoolerNormalize(),
        )

        # Create token pooler with AllPool method
        token_embed_pooler = TokenPooler(
            pooling=AllPool(),
            head=token_embed_head,
        )

        # Create token classify pooler
        token_classify_pooler = pooler_for_token_classify(
            pooler_config,
            pooling=AllPool(),
            classifier=self.sparse_linear,
            act_fn=torch.relu,
        )

        # Wrap with BOS/EOS filter
        token_embed_pooler = SpecialTokenFilterPooler(
            token_embed_pooler,
            [self.bos_token_id],
        )
        token_classify_pooler = SpecialTokenFilterPooler(
            token_classify_pooler,
            [self.bos_token_id, self.eos_token_id],
        )

        return DispatchPooler(
            {
                "embed": pooler_for_embed(pooler_config),
                "token_embed": token_embed_pooler,
                "token_classify": token_classify_pooler,
            }
        )

    def load_weights(self, all_weights: Iterable[tuple[str, torch.Tensor]]):
        secondary_weights, primary_weights = filter_secondary_weights(
            all_weights,
            self.secondary_weight_prefixes,
        )

        super().load_weights(primary_weights)

        params_dict = dict(self.named_parameters())
        for name, loaded_weight in secondary_weights:
            if name not in params_dict:
                continue
            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)


class SpecialTokenFilterPooler(Pooler):
    """Wraps a pooler and filters BOS/EOS tokens from token-level outputs."""

    def __init__(self, pooler: Pooler, token_ids_to_skip: list[int | None]) -> None:
        super().__init__()
        self.pooler = pooler
        self.token_ids_to_skip = tuple(
            token_id for token_id in token_ids_to_skip if token_id is not None
        )

    def get_supported_tasks(self) -> Set[PoolingTask]:
        return self.pooler.get_supported_tasks()

    def get_pooling_updates(self, task: PoolingTask) -> PoolingParamsUpdate:
        return PoolingParamsUpdate(requires_token_ids=True)

    def _filter_one(
        self,
        data: torch.Tensor | None,
        token_ids: torch.Tensor,
    ) -> torch.Tensor | None:
        if data is None:
            return None

        keep_mask = torch.ones_like(token_ids, dtype=torch.bool)
        for token_id in self.token_ids_to_skip:
            keep_mask &= token_ids != token_id
        return data[keep_mask]

    def forward(
        self,
        hidden_states: torch.Tensor | list[torch.Tensor],
        pooling_metadata: PoolingMetadata,
    ) -> PoolerOutput:
        outputs = self.pooler(hidden_states, pooling_metadata)
        prompt_token_ids = pooling_metadata.get_prompt_token_ids()
        return [
            self._filter_one(output, token_ids)
            for output, token_ids in zip(outputs, prompt_token_ids)
        ]
