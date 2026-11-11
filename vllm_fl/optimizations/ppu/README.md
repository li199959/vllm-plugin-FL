# T-Head PPU model optimizations

These opt-in modules target vLLM 0.24 and the T-Head builds of PyTorch,
Triton and FlagGems. Enable only the switches for the model being served;
registration skips other vendors. Install the plugin before running the
commands below. Replace `/path/to/model` with a local checkpoint.

## MiniMax-M3

The hybrid MoE configuration targets BF16 TP16 shards with 128 experts,
top-4 routing, hidden size 6144 and intermediate size 192 per rank.
Decode batches up to 128 tokens use PPU DeepGEMM with its small-K GEMV
path disabled; larger batches use tuned FlagGems kernels.

```bash
VLLM_FL_MOE=hybrid \
VLLM_FL_TOPK_DISPATCH=1 \
VLLM_FL_M3_DENSE=hybrid \
VLLM_FL_M3_GEMMA_NORM=flaggems \
VLLM_FL_M3_FUSED_IMPL=triton \
VLLM_FL_FLAGOS_BLACKLIST=index_copy_,index,linear,addmm,mm,empty.memory_format \
vllm serve /path/to/model --tensor-parallel-size 16 --block-size 128 \
  --max-model-len 32768 --trust-remote-code
```

The Triton fused Q/K norm, RoPE and KV insertion path supports `auto`
FP16/BF16 KV caches. The dense override uses split-K or explicit FlagGems
tiles for small batches and preserves the native large-batch fallback.
Keep `linear,addmm,mm` blacklisted when using that fallback.

The optional native M3 extension is built from source with the PPU SDK;
the repository does not supply a prebuilt shared library:

```bash
VLLM_VENDOR=thead python -m pip wheel . --no-deps --no-build-isolation -w /tmp/ppu-wheels
```

The build uses the active Python/PyTorch ABI and detects the visible PPU.
For a build host without a PPU, set `VLLM_FL_PPU_ARCH=compute_80a,sm_80a`
or `compute_89,sm_89` to match the target hardware. `CUDA_HOME` selects the
vendor SDK. Without this extension the plugin retains its Python/Triton
fallback. `VLLM_FL_M3_FUSED_IMPL=triton` explicitly selects Triton even when
the native extension is installed.

## Qwen3-VL-30B-A3B

The tuned dense and MoE tiles target TP2 BF16 text layers. FA3 modes require
the T-Head `flash_attn_3` wheel; hybrid MoE additionally requires PPU DeepGEMM.

| Switch | Behavior |
|---|---|
| `VLLM_FL_QWEN3VL_STABLE=1` | Use shared shape-stable index, copy and MoE alignment kernels. |
| `VLLM_FL_QWEN3VL_DENSE=tuned` | Use explicit tiles for supported decode projections. |
| `VLLM_FL_QWEN3VL_MOE=tuned` | Use FlagGems MoE tile tables. |
| `VLLM_FL_QWEN3VL_MOE=hybrid` | Use DeepGEMM at `VLLM_FL_QWEN3VL_DEEPGEMM_MIN` tokens (default 8192), FlagGems below it. |
| `VLLM_FL_QWEN3VL_ATTN=tuned` | Tune Triton decode launch parameters. |
| `VLLM_FL_QWEN3VL_ATTN=hybrid` | Use FA3 for eligible prefill/mixed steps, tuned Triton for decode. |
| `VLLM_FL_QWEN3VL_ATTN=fa3` | Use FA3 for all eligible attention steps. |

Serve with `vllm serve /path/to/model --tensor-parallel-size 2` after setting
the desired switches. With FlagGems versions affected by small-image
head-dimension-72 SDPA errors, append
`scaled_dot_product_attention,_scaled_dot_product_flash_attention,_flash_attention_forward`
to `VLLM_FL_FLAGOS_BLACKLIST_APPEND`.

## BGE-M3 and Qwen3-VL-Embedding-8B

These configurations target a single-device pooling runner:
`vllm serve /path/to/model --runner pooling`.

| Model | Switches |
|---|---|
| BGE-M3 (FP16) | `VLLM_FL_BGEM3_STABLE=1`, `VLLM_FL_BGEM3_DENSE=hybrid`; select projections with `VLLM_FL_BGEM3_GEMS_LINEARS` (`qkv,o,fc1,fc2`, default `o`). `VLLM_FL_BGEM3_GEMS_SMALL_M` enables all projections below a token threshold (default 0). |
| BGE-M3 attention | `VLLM_FL_BGEM3_ATTN=fa3` uses the PPU FA3 wheel above `VLLM_FL_BGEM3_FA3_MIN_TOKENS` (default 2048). |
| Qwen3-VL-Embedding-8B (BF16) | `VLLM_FL_Q3VLEMB_STABLE=1`, `VLLM_FL_Q3VLEMB_DENSE=hybrid`, and a positive `VLLM_FL_Q3VLEMB_GEMS_SMALL_M` select tuned small-batch linears. The default threshold 0 disables them. |

For hybrid dense modes append `mm,mm_out,addmm,addmm_out,linear` to
`VLLM_FL_FLAGOS_BLACKLIST_APPEND` so larger batches retain the native path.
BGE-M3 also needs `broadcast_to` blacklisted on FlagGems versions that
perform a host-to-device copy in that operator during graph capture.
For Qwen3-VL-Embedding use the SDPA exclusions listed above when needed.
Embedding requests must use the checkpoint's conversation template,
`add_special_tokens=true`, and `add_generation_prompt=true` for chat input
so last-token pooling selects the intended token.

## Whisper

Whisper-specific compilation patches apply to `WhisperModel`; models that
only reuse `WhisperEncoder` keep their original encoder implementation.

```bash
VLLM_FL_WHISPER_COMPILE_ENCODER=1 \
VLLM_FL_WHISPER_ENCODER_GRAPH_MAX=8 \
VLLM_FL_WHISPER_COMPILE_PREFILL=1 \
VLLM_FL_WHISPER_CROSS_SLOTS=1 \
VLLM_FL_FLAGOS_BLACKLIST_APPEND=broadcast_to,mm,mm_out,addmm,addmm_out,linear,empty \
vllm serve /path/to/model --served-model-name whisper-large-v3
```

`VLLM_FL_WHISPER_ENCODER_GRAPH_MAX` defaults to 0 (no encoder graph capture)
and requires encoder compilation. `VLLM_FL_WHISPER_STABLE=1` additionally
enables shared index/copy kernels. The cross-slot mapping switch replaces
the shared encoder-decoder slot builder with an equivalent batched gather.

## Unlimited-OCR

Model/config registration and R-SWA attention are independent of the PPU
optimization switch. `VLLM_FL_UOCR_STABLE=1` enables shared shape-stable
FlagGems kernels. R-SWA requires the Triton attention backend and disables
prefix caching. Serve with `vllm serve /path/to/model --trust-remote-code`.

## Development

Shared launchers live in `common/`; each model owns its supported shapes,
tiles and environment switches. Add model packages to `ppu._MODELS`.
Regression tests are under `tests/unit_tests/optimizations/` and
`tests/functional_tests/optimizations/ppu/`. PPU functional tests are gated
before import on other vendors. Use the standard runner:

```bash
python tests/run.py --platform thead --device ppu --scope functional
```
