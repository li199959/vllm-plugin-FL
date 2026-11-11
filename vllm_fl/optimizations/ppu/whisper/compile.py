# Copyright (c) 2026 BAAI. All rights reserved.
"""Compiled encoder stack and compiled-decoder prefill for vLLM's WhisperModel.

Both patches are installed on ``WhisperModel`` methods only, so models that reuse
``WhisperEncoder`` / ``WhisperDecoder`` elsewhere keep vLLM's code paths.
"""

import logging

import torch
from torch import nn

logger = logging.getLogger(__name__)

_installed = False


# --- encoder -----------------------------------------------------------------
# WhisperEncoder.forward = per-audio conv1/conv2/gelu + positional add (cheap, kept
# eager), then 32 transformer layers + final LayerNorm over [num_audios, 1500, 1280].
# Only the layer stack is compiled, as one graph; dynamo's automatic dynamic shapes
# turn the batch dim symbolic after the first recompile, so a few graphs serve every
# number of audios per step.
#
# The attention core must not break the graph: vllm::torch_sdpa_wrapper's fake impl
# rejects the call dynamo makes (TypeError: missing 'scale' and 'cu_seqlens'), and at
# every graph break dynamo also grabs the resume frames of FlagGems' Python kernels
# (layer_norm etc.) and inlines their Triton launches, which inductor then fails to
# build (NameError: 'ext'). ``fl_whisper::encoder_sdpa`` is the same computation
# (b s h d -> SDPA -> b s (h d), Gems flash kernel via aten) with a working fake impl.

_encoder_sdpa_op = None


def _register_encoder_sdpa():
    global _encoder_sdpa_op
    if _encoder_sdpa_op is not None:
        return _encoder_sdpa_op

    @torch.library.custom_op("fl_whisper::encoder_sdpa", mutates_args=())
    def encoder_sdpa(
        q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, num_heads: int, scale: float
    ) -> torch.Tensor:
        b, s, _ = q.shape
        q, k, v = (t.view(b, s, num_heads, -1).transpose(1, 2) for t in (q, k, v))
        out = nn.functional.scaled_dot_product_attention(
            q, k, v, dropout_p=0.0, scale=scale
        )
        return out.transpose(1, 2).reshape(b, s, -1)

    @encoder_sdpa.register_fake
    def _(q, k, v, num_heads, scale):
        return torch.empty_like(q)

    _encoder_sdpa_op = encoder_sdpa
    return encoder_sdpa


def _sdpa_forward(attn, op):
    def forward(query, key, value, *args, **kwargs):
        return op(query, key, value, attn.num_heads, float(attn.scale))

    return forward


def _encoder_stack(encoder, hidden_states):
    for layer in encoder.layers:
        hidden_states = layer(hidden_states)
    return encoder.layer_norm(hidden_states)


def _compiled_stack(encoder):
    fn = getattr(encoder, "_fl_compiled_stack", None)
    if fn is None:
        op = _register_encoder_sdpa()
        for layer in encoder.layers:
            attn = layer.self_attn.attn  # WhisperEncoderAttention (MMEncoderAttention)
            assert attn.num_heads == attn.num_kv_heads, "encoder attention has no GQA"
            attn.forward = _sdpa_forward(attn, op)
        # dynamic=None: first shape static, later batch sizes recompile once as dynamic
        fn = torch.compile(_encoder_stack, backend="inductor", fullgraph=True)
        encoder._fl_compiled_stack = fn
    return fn


# The compiled stack still launches ~10 kernels per layer from Python (~33 ms CPU
# for 32 layers), more than the GPU needs for a few audios. Batches of at most
# VLLM_FL_WHISPER_ENCODER_GRAPH_MAX audios replay a CUDA graph captured on first use
# (one per batch size, shared memory pool); larger batches are GPU bound and run the
# compiled stack directly. Outputs are cloned: vLLM caches encoder outputs across
# steps, and the next replay overwrites the static buffer.


class _EncoderGraphs:
    def __init__(self, fn, max_batch):
        self.fn, self.max_batch = fn, max_batch
        self.graphs = {}
        self.pool = None

    def _capture(self, encoder, hidden_states):
        static_in = hidden_states.clone()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(2):  # settle compile / autotune outside capture
                self.fn(encoder, static_in)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        if self.pool is None:
            self.pool = torch.cuda.graph_pool_handle()
        with torch.cuda.graph(graph, pool=self.pool):
            static_out = self.fn(encoder, static_in)
        return graph, static_in, static_out

    def __call__(self, encoder, hidden_states):
        n = hidden_states.size(0)
        if n > self.max_batch or torch.cuda.is_current_stream_capturing():
            return self.fn(encoder, hidden_states)
        entry = self.graphs.get(n)
        if entry is None:
            entry = self.graphs[n] = self._capture(encoder, hidden_states)
        graph, static_in, static_out = entry
        static_in.copy_(hidden_states)
        graph.replay()
        return static_out.clone()


def _encoder_runner(encoder):
    run = getattr(encoder, "_fl_encoder_runner", None)
    if run is None:
        import os

        fn = _compiled_stack(encoder)
        max_batch = int(os.environ.get("VLLM_FL_WHISPER_ENCODER_GRAPH_MAX", "0") or 0)
        run = _EncoderGraphs(fn, max_batch) if max_batch > 0 else fn
        encoder._fl_encoder_runner = run
    return run


def _encoder_forward(encoder, input_features):
    hidden_states = []
    input_is_batched = False
    for features in input_features:
        embeds = nn.functional.gelu(encoder.conv1(features))
        embeds = nn.functional.gelu(encoder.conv2(embeds))
        embeds = embeds.transpose(-1, -2)
        embeds = (embeds + encoder.embed_positions.weight[: embeds.size(-2), :]).to(
            embeds.dtype
        )
        hidden_states.append(embeds)
        input_is_batched = embeds.ndim > 2
    if input_is_batched:
        hidden_states = torch.cat(hidden_states)
    else:
        hidden_states = torch.stack(hidden_states, dim=0)
    return _encoder_runner(encoder)(encoder, hidden_states)


def _get_encoder_outputs(self, input_features):
    if input_features is None:
        return None
    return _encoder_forward(self.encoder, input_features)


# --- prefill -----------------------------------------------------------------
# vLLM sets skip_compiled on any step with encoder inputs because the decoder graph
# is traced with encoder_hidden_states=None (cross K/V read from the cache). Writing
# the new requests' cross K/V into the cache here, exactly as CrossAttentionImpl
# does (builder's cross slot_mapping), lets those steps reuse the same graph.


def _write_cross_kv(decoder, enc_states):
    from vllm.model_executor.layers.attention.attention import get_attention_context

    for layer in decoder.layers:
        cross = layer.encoder_attn
        attn = cross.attn
        attn_metadata, attn_layer, kv_cache, _ = get_attention_context(attn.layer_name)
        if attn_metadata is None:
            continue  # profile / dummy run: nothing to cache
        kv, _ = cross.kv_proj(enc_states)
        k, v = kv.split([cross.kv_size, cross.kv_size], dim=-1)
        attn_layer.impl.do_kv_cache_update(
            attn_layer,
            k.view(-1, attn.num_kv_heads, attn.head_size),
            v.view(-1, attn.num_kv_heads, attn.head_size_v),
            kv_cache,
            attn_metadata.slot_mapping,
        )


def _model_forward(self, input_ids, positions, encoder_outputs):
    from vllm.forward_context import get_forward_context, is_forward_context_available

    if not len(encoder_outputs) or not is_forward_context_available():
        return self._fl_orig_forward(input_ids, positions, encoder_outputs)
    ctx = get_forward_context()
    if ctx.attn_metadata is None:
        return self._fl_orig_forward(input_ids, positions, encoder_outputs)
    _write_cross_kv(self.decoder, torch.cat(encoder_outputs, dim=0))
    skip = ctx.skip_compiled
    ctx.skip_compiled = False
    try:
        return self.decoder(
            input_ids=input_ids, positions=positions, encoder_hidden_states=None
        )
    finally:
        ctx.skip_compiled = skip


def install(encoder: bool, prefill: bool) -> None:
    global _installed
    if _installed:
        return
    from vllm.model_executor.models.whisper import WhisperModel

    if encoder:
        WhisperModel.get_encoder_outputs = _get_encoder_outputs
    if prefill:
        WhisperModel._fl_orig_forward = WhisperModel.forward
        WhisperModel.forward = _model_forward
    _installed = True
    logger.info(
        "Whisper optimizations: compile_encoder=%s compile_prefill=%s", encoder, prefill
    )
