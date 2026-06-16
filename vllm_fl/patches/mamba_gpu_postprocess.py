# Copyright (c) 2025 BAAI. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU-side Mamba/GDN align-cache postprocess for hybrid spec decode.

This backports the core idea of vLLM PR #40172 into the plugin runner:
when speculative decoding runs on hybrid Mamba/GDN models with
``mamba_cache_mode == "align"``, avoid the per-step GPU->CPU sync used to
decide/copy accepted speculative states.
"""

from __future__ import annotations

import dataclasses
import os
from collections.abc import Callable
from typing import Any

import torch

from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateCopyFunc,
    get_conv_copy_spec,
    get_temporal_copy_spec,
)
from vllm.triton_utils import tl, triton
from vllm.v1.kv_cache_interface import KVCacheConfig, MambaSpec
from vllm.v1.utils import CpuGpuBuffer
from vllm.v1.worker.gpu_input_batch import CachedRequestState
from vllm.v1.worker.lora_model_runner_mixin import GPUInputBatch
from vllm.v1.worker import mamba_utils
from vllm.v1.core.sched.output import SchedulerOutput


def enable_mamba_gpu_postprocess() -> bool:
    return os.environ.get(
        "VLLM_FL_ENABLE_MAMBA_GPU_POSTPROCESS", "1"
    ).lower() not in ("0", "false", "no", "off")


@triton.jit
def _postprocess_mamba_fused_kernel(
    num_accepted_tokens_ptr,
    mamba_state_idx_ptr,
    num_scheduled_tokens_ptr,
    num_computed_tokens_ptr,
    num_draft_tokens_ptr,
    block_table_ptrs_ptr,
    block_table_stride_req: tl.int64,
    state_base_addrs_ptr,
    state_block_strides_ptr,
    state_elem_sizes_ptr,
    state_inner_sizes_ptr,
    state_conv_widths_ptr,
    state_group_indices_ptr,
    num_accepted_tokens_out_ptr,
    num_reqs,
    block_size: tl.constexpr,
    COPY_BLOCK_SIZE: tl.constexpr,
):
    req_idx = tl.program_id(0)
    state_idx = tl.program_id(1)
    if req_idx >= num_reqs:
        return

    num_accepted = tl.load(num_accepted_tokens_ptr + req_idx)
    src_block_idx = tl.load(mamba_state_idx_ptr + req_idx)
    num_scheduled = tl.load(num_scheduled_tokens_ptr + req_idx)
    num_computed = tl.load(num_computed_tokens_ptr + req_idx)
    num_draft = tl.load(num_draft_tokens_ptr + req_idx)

    num_tokens_running_state = num_computed + num_scheduled - num_draft
    new_num_computed = num_tokens_running_state + num_accepted - 1
    aligned_new_computed = (new_num_computed // block_size) * block_size
    if aligned_new_computed < num_tokens_running_state:
        return

    accept_token_bias = aligned_new_computed - num_tokens_running_state
    dest_block_idx = aligned_new_computed // block_size - 1

    state_base_addr = tl.load(state_base_addrs_ptr + state_idx)
    state_block_stride = tl.load(state_block_strides_ptr + state_idx)
    state_elem_size = tl.load(state_elem_sizes_ptr + state_idx)
    state_inner_size = tl.load(state_inner_sizes_ptr + state_idx)
    conv_width = tl.load(state_conv_widths_ptr + state_idx)
    group_idx = tl.load(state_group_indices_ptr + state_idx).to(tl.int64)

    group_base_addr = tl.load(block_table_ptrs_ptr + group_idx)
    block_table_typed = group_base_addr.to(tl.pointer_type(tl.int32))
    block_table_base = block_table_typed + req_idx * block_table_stride_req

    src_block_id = tl.load(block_table_base + src_block_idx).to(tl.int64)
    dest_block_id = tl.load(block_table_base + dest_block_idx).to(tl.int64)

    if conv_width > 0:
        src_offset = accept_token_bias.to(tl.int64) * state_inner_size * state_elem_size
        src_addr = state_base_addr + src_block_id * state_block_stride + src_offset
        dst_addr = state_base_addr + dest_block_id * state_block_stride
        copy_size = (
            (conv_width - accept_token_bias).to(tl.int64)
            * state_inner_size
            * state_elem_size
        )
    else:
        actual_src_block_idx = src_block_idx + accept_token_bias
        actual_src_block_id = tl.load(block_table_base + actual_src_block_idx).to(
            tl.int64
        )
        src_addr = state_base_addr + actual_src_block_id * state_block_stride
        dst_addr = state_base_addr + dest_block_id * state_block_stride
        copy_size = state_inner_size * state_elem_size

    if src_block_idx == dest_block_idx and state_idx == 0:
        tl.store(num_accepted_tokens_out_ptr + req_idx, 1)

    if src_block_idx == dest_block_idx and accept_token_bias == 0:
        return

    offsets = tl.arange(0, COPY_BLOCK_SIZE)
    for i in range(0, copy_size, COPY_BLOCK_SIZE):
        mask = (i + offsets) < copy_size
        curr_src = (src_addr + i + offsets).to(tl.pointer_type(tl.uint8))
        curr_dst = (dst_addr + i + offsets).to(tl.pointer_type(tl.uint8))
        data = tl.load(curr_src, mask=mask)
        tl.store(curr_dst, data, mask=mask)


def _get_mamba_groups(kv_cache_config: KVCacheConfig) -> tuple[list[int], MambaSpec]:
    mamba_group_ids: list[int] = []
    mamba_specs: list[MambaSpec] = []
    for i, group in enumerate(kv_cache_config.kv_cache_groups):
        spec = group.kv_cache_spec
        if isinstance(spec, MambaSpec):
            mamba_group_ids.append(i)
            mamba_specs.append(spec)
    assert mamba_group_ids, "no mamba layers in the model"
    assert all(mamba_specs[0] == spec for spec in mamba_specs)
    return mamba_group_ids, mamba_specs[0]


@dataclasses.dataclass
class MambaSpecDecodeGPUContext:
    state_base_addrs: torch.Tensor
    state_block_strides: torch.Tensor
    state_elem_sizes: torch.Tensor
    state_inner_sizes: torch.Tensor
    state_conv_widths: torch.Tensor
    state_group_indices: torch.Tensor
    block_size: int
    num_layers: int
    num_state_types: int
    mamba_group_ids: list[int]
    num_groups: int
    num_accepted_tokens_out: torch.Tensor
    block_table_ptrs: torch.Tensor
    mamba_state_idx_buf: CpuGpuBuffer
    num_scheduled_tokens_buf: CpuGpuBuffer
    num_computed_tokens_buf: CpuGpuBuffer
    num_draft_tokens_buf: CpuGpuBuffer
    block_table_stride_req: int = 0
    is_initialized: bool = False

    @classmethod
    def create(
        cls,
        max_num_reqs: int,
        kv_cache_config: KVCacheConfig,
        num_state_types: int,
        device: torch.device,
        make_buffer: Callable[..., CpuGpuBuffer],
    ) -> "MambaSpecDecodeGPUContext":
        mamba_group_ids, mamba_spec = _get_mamba_groups(kv_cache_config)
        num_layers = sum(
            len(kv_cache_config.kv_cache_groups[gid].layer_names)
            for gid in mamba_group_ids
        )
        total_states = num_layers * num_state_types
        return cls(
            state_base_addrs=torch.zeros(total_states, dtype=torch.int64, device=device),
            state_block_strides=torch.zeros(
                total_states, dtype=torch.int64, device=device
            ),
            state_elem_sizes=torch.zeros(total_states, dtype=torch.int32, device=device),
            state_inner_sizes=torch.zeros(
                total_states, dtype=torch.int64, device=device
            ),
            state_conv_widths=torch.zeros(
                total_states, dtype=torch.int32, device=device
            ),
            state_group_indices=torch.zeros(
                total_states, dtype=torch.int32, device=device
            ),
            block_size=mamba_spec.block_size,
            num_layers=num_layers,
            num_state_types=num_state_types,
            mamba_group_ids=mamba_group_ids,
            num_groups=len(mamba_group_ids),
            num_accepted_tokens_out=torch.zeros(
                max_num_reqs, dtype=torch.int32, device=device
            ),
            block_table_ptrs=torch.zeros(
                len(mamba_group_ids), dtype=torch.int64, device=device
            ),
            mamba_state_idx_buf=make_buffer(max_num_reqs, dtype=torch.int32),
            num_scheduled_tokens_buf=make_buffer(max_num_reqs, dtype=torch.int32),
            num_computed_tokens_buf=make_buffer(max_num_reqs, dtype=torch.int32),
            num_draft_tokens_buf=make_buffer(max_num_reqs, dtype=torch.int32),
        )

    def initialize_from_forward_context(
        self,
        kv_cache_config: KVCacheConfig,
        forward_context: dict[str, Any],
        mamba_state_copy_funcs: tuple[MambaStateCopyFunc, ...],
        block_tables: list[torch.Tensor],
    ) -> None:
        if self.is_initialized:
            return

        idx = 0
        for group_local_idx, mamba_group_id in enumerate(self.mamba_group_ids):
            layer_names = kv_cache_config.kv_cache_groups[mamba_group_id].layer_names
            for layer_name in layer_names:
                attention = forward_context[layer_name]
                kv_caches: list[torch.Tensor] = attention.kv_cache
                for state_type_idx, state in enumerate(kv_caches):
                    self.state_base_addrs[idx] = state.data_ptr()
                    block_stride_elems = state.stride(0) if state.dim() > 1 else state.numel()
                    self.state_block_strides[idx] = (
                        block_stride_elems * state.element_size()
                    )
                    self.state_elem_sizes[idx] = state.element_size()

                    copy_func = mamba_state_copy_funcs[state_type_idx]
                    assert copy_func in (get_conv_copy_spec, get_temporal_copy_spec), (
                        f"unexpected copy func: {copy_func}"
                    )
                    if copy_func is get_conv_copy_spec:
                        self.state_conv_widths[idx] = state.size(1) if state.dim() > 1 else 0
                        self.state_inner_sizes[idx] = state.stride(1) if state.dim() > 2 else 1
                    else:
                        self.state_conv_widths[idx] = 0
                        self.state_inner_sizes[idx] = state[0].numel() if state.dim() > 1 else 1

                    self.state_group_indices[idx] = group_local_idx
                    idx += 1

        assert len(block_tables) == self.num_groups
        strides = {bt.stride(0) for bt in block_tables}
        assert len(strides) == 1
        self.block_table_stride_req = int(next(iter(strides)))
        for i, bt in enumerate(block_tables):
            self.block_table_ptrs[i] = bt.data_ptr()
        self.is_initialized = True

    def run(
        self,
        num_reqs: int,
        num_accepted_tokens_gpu: torch.Tensor,
    ) -> None:
        if num_reqs == 0 or not self.is_initialized:
            return
        self.num_accepted_tokens_out[:num_reqs].copy_(
            num_accepted_tokens_gpu[:num_reqs]
        )
        total_states = self.num_layers * self.num_state_types
        _postprocess_mamba_fused_kernel[(num_reqs, total_states)](
            num_accepted_tokens_gpu,
            self.mamba_state_idx_buf.gpu,
            self.num_scheduled_tokens_buf.gpu,
            self.num_computed_tokens_buf.gpu,
            self.num_draft_tokens_buf.gpu,
            self.block_table_ptrs,
            self.block_table_stride_req,
            self.state_base_addrs,
            self.state_block_strides,
            self.state_elem_sizes,
            self.state_inner_sizes,
            self.state_conv_widths,
            self.state_group_indices,
            self.num_accepted_tokens_out,
            num_reqs,
            block_size=self.block_size,
            COPY_BLOCK_SIZE=1024,
        )


def create_context(
    *,
    max_num_reqs: int,
    kv_cache_config: KVCacheConfig,
    copy_funcs: tuple[MambaStateCopyFunc, ...],
    device: torch.device,
    make_buffer: Callable[..., CpuGpuBuffer],
) -> MambaSpecDecodeGPUContext:
    return MambaSpecDecodeGPUContext.create(
        max_num_reqs=max_num_reqs,
        kv_cache_config=kv_cache_config,
        num_state_types=len(copy_funcs),
        device=device,
        make_buffer=make_buffer,
    )


def stage_inputs_to_gpu(
    *,
    ctx: MambaSpecDecodeGPUContext,
    scheduler_output: SchedulerOutput,
    req_ids: list[str],
    num_reqs: int,
    requests: dict[str, CachedRequestState],
    mamba_state_idx: dict[str, int],
) -> None:
    state_np = ctx.mamba_state_idx_buf.np
    scheduled_np = ctx.num_scheduled_tokens_buf.np
    computed_np = ctx.num_computed_tokens_buf.np
    draft_np = ctx.num_draft_tokens_buf.np
    scheduled = scheduler_output.num_scheduled_tokens
    scheduled_spec = scheduler_output.scheduled_spec_decode_tokens

    for i in range(num_reqs):
        req_id = req_ids[i]
        state_idx = mamba_state_idx.get(req_id)
        assert state_idx is not None, (
            f"mamba_state_idx missing entry for {req_id!r}; "
            "preprocess_mamba must run before GPU postprocess staging"
        )
        state_np[i] = state_idx
        scheduled_np[i] = scheduled[req_id]
        computed_np[i] = requests[req_id].num_computed_tokens
        draft_np[i] = len(scheduled_spec.get(req_id, []))

    ctx.mamba_state_idx_buf.copy_to_gpu(num_reqs)
    ctx.num_scheduled_tokens_buf.copy_to_gpu(num_reqs)
    ctx.num_computed_tokens_buf.copy_to_gpu(num_reqs)
    ctx.num_draft_tokens_buf.copy_to_gpu(num_reqs)


def postprocess_align_gpu(
    *,
    ctx: MambaSpecDecodeGPUContext,
    num_reqs: int,
    num_accepted_tokens_gpu: torch.Tensor,
    num_accepted_tokens_cpu_tensor: torch.Tensor,
    input_batch: GPUInputBatch,
    kv_cache_config: KVCacheConfig,
    forward_context: dict[str, Any],
    mamba_state_copy_funcs: tuple[MambaStateCopyFunc, ...],
) -> None:
    if not ctx.is_initialized:
        ctx.initialize_from_forward_context(
            kv_cache_config,
            forward_context,
            mamba_state_copy_funcs,
            [
                input_batch.block_table[gid].get_device_tensor(num_reqs)
                for gid in ctx.mamba_group_ids
            ],
        )

    ctx.run(num_reqs=num_reqs, num_accepted_tokens_gpu=num_accepted_tokens_gpu)
    num_accepted_tokens_cpu_tensor[:num_reqs].copy_(
        ctx.num_accepted_tokens_out[:num_reqs], non_blocking=True
    )


def postprocess_align_cpu_fallback(
    *,
    scheduler_output: SchedulerOutput,
    kv_cache_config: KVCacheConfig,
    input_batch: GPUInputBatch,
    requests: dict[str, CachedRequestState],
    mamba_state_idx: dict[str, int],
    forward_context: dict[str, Any],
    mamba_state_copy_funcs: tuple[MambaStateCopyFunc, ...],
    copy_bufs: mamba_utils.MambaCopyBuffers,
    num_reqs: int,
    num_accepted_tokens_gpu: torch.Tensor,
) -> None:
    for i, num_tokens in enumerate(num_accepted_tokens_gpu[:num_reqs].cpu().numpy()):
        input_batch.num_accepted_tokens_cpu[i] = num_tokens
    mamba_utils.postprocess_mamba(
        scheduler_output,
        kv_cache_config,
        input_batch,
        requests,
        mamba_state_idx,
        forward_context,
        mamba_state_copy_funcs,
        copy_bufs,
    )
