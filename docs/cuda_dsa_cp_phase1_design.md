# CUDA DSA-CP 阶段一设计说明

## 1. 背景

DeepSeek-V3.2 使用 MLA 和 DSA（DeepSeek Sparse Attention）。在 vLLM CUDA 路径中，MLA 主体已经大量使用 Tensor Parallel：

- `q_b_proj` 使用 `ColumnParallelLinear`
- `kv_b_proj` 使用 `ColumnParallelLinear`
- `o_proj` 使用 `RowParallelLinear`
- sparse attention 由 vLLM 原生 `FlashMLA Sparse` / `SparseAttnIndexer` 路径负责

因此，Ascend DSA-CP 中面向 `q_b_proj`、`o_proj` 的 layer sharding 设计不能直接带来同等收益。CUDA 阶段一选择了一个更保守的切入点：只对 DeepSeek MLA 中仍然复制计算的 `fused_qkv_a_proj` 做 token 维度并行，其他 attention、indexer、KV cache、FlashMLA sparse 路径保持原生 vLLM 行为。

阶段一目标不是最终完整 DSA-CP，而是先确认：

- CUDA 插件可以安全接管 DeepSeek-V3.2 MLA wrapper
- token 维度切分与 TP 通信链路可行
- 保持原生 sparse attention 语义不变
- 保持 CUDA graph 兼容

## 2. 普通 vLLM 推理路径

普通 DeepSeek-V3.2 MLA 推理中，每个 TP rank 都会完整处理当前 batch 的全部 token。

以一个 MLA layer 为例，普通路径可以简化为：

```text
hidden_states
  -> fused_qkv_a_proj
  -> split q_c / kv_lora
  -> q_a_layernorm
  -> q_b_proj
  -> kv_a_layernorm
  -> rotary embedding
  -> sparse indexer
  -> FlashMLA Sparse attention
  -> o_proj
```

其中：

```text
fused_qkv_a_proj: 每个 TP rank 都对全部 token 计算一遍
q_b_proj:         CUDA 上已经按 TP 切分
kv_b_proj:        CUDA 上已经按 TP 切分
o_proj:           CUDA 上已经按 TP 切分
sparse indexer:   保持 vLLM 原生完整 token 输入
FlashMLA Sparse:  保持 vLLM 原生 attention backend
```

普通路径的特点是实现成熟、CUDA graph 兼容好，但 `fused_qkv_a_proj` 在 TP ranks 间存在重复 token 计算。

## 3. 阶段一设计

阶段一只改变 `fused_qkv_a_proj` 的执行方式。

普通路径：

```text
每个 rank:
  fused_qkv_a_proj(all tokens)
```

阶段一路径：

```text
每个 rank:
  只取自己负责的一段 token
  fused_qkv_a_proj(local tokens)
  padding 到统一长度
  tensor_model_parallel_all_gather
  还原为完整 token 顺序
```

也就是说，阶段一把 `fused_qkv_a_proj` 从“每卡重复算全部 token”改成“每卡只算一段 token，然后 all-gather 拼回完整结果”。

核心数据流如下：

```text
hidden_states: [num_tokens, hidden_size]
        |
        | token split by TP rank
        v
local_hidden_states: [ceil(num_tokens / tp_size), hidden_size]
        |
        | fused_qkv_a_proj
        v
local_qkv_lora: [ceil(num_tokens / tp_size), q_lora_rank + kv_lora_rank + rope_dim]
        |
        | all_gather along token dim
        v
qkv_lora: [padded_tokens, q_lora_rank + kv_lora_rank + rope_dim]
        |
        | remove padding
        v
qkv_lora[:num_tokens]
```

之后的计算继续沿用普通 vLLM 逻辑：

```text
qkv_lora
  -> split q_c / kv_lora
  -> q_a_layernorm
  -> q_b_proj
  -> kv_a_layernorm
  -> rotary embedding
  -> sparse indexer
  -> FlashMLA Sparse attention
  -> o_proj
```

## 4. 为什么阶段一功能安全

阶段一只改变 `fused_qkv_a_proj` 的计算分布，不改变它的数学结果。

原始计算可以表示为：

```text
qkv_lora = fused_qkv_a_proj(hidden_states)
```

阶段一等价于：

```text
local_qkv_lora_i = fused_qkv_a_proj(hidden_states_i)
qkv_lora = concat(all_gather(local_qkv_lora_i))
```

由于 `fused_qkv_a_proj` 是逐 token 的 Linear，不依赖 token 之间的信息，因此按 token 切分后再拼回，与一次性计算全部 token 在数学上等价。

阶段一没有修改：

- position / rotary embedding 语义
- sparse indexer 的输入语义
- KV cache 写入逻辑
- top-k sparse metadata
- FlashMLA sparse attention kernel
- `q_b_proj`、`kv_b_proj`、`o_proj` 的 TP 行为

因此阶段一的风险集中在 token 切分、padding、all-gather 和顺序恢复上，语义边界比较清晰。

## 5. 与普通推理的具体区别

| 模块 | 普通 vLLM CUDA 路径 | 阶段一路径 |
| --- | --- | --- |
| `fused_qkv_a_proj` | 每个 TP rank 处理全部 token | 每个 TP rank 只处理一段 token |
| token 维度通信 | 无额外通信 | `fused_qkv_a_proj` 后增加一次 all-gather |
| `q_b_proj` | 原生 TP sharded | 不变 |
| `kv_b_proj` | 原生 TP sharded | 不变 |
| `o_proj` | 原生 TP sharded | 不变 |
| sparse indexer | 原生完整 token 路径 | 不变 |
| FlashMLA Sparse | 原生 backend | 不变 |
| CUDA graph | 原生支持 | 保持支持 |
| 数学结果 | 原生结果 | 预期等价 |

## 6. 代码位置

阶段一主要实现位于：

```text
vllm_fl/ops/dsa_cp_mla.py
```

关键逻辑：

- `CudaDSACPMultiHeadLatentAttentionWrapper`
  - 替换 vLLM 原生 `MultiHeadLatentAttentionWrapper`
  - 负责读取开关、判断 sparse model、TP size、mode

- `_fused_qkv_a_proj_token_parallel`
  - 对 `hidden_states` 按 token 维度切分
  - 调用 `fused_qkv_a_proj(local_hidden_states)`
  - 对最后一个 rank 的不足 token 进行 padding
  - 调用 `tensor_model_parallel_all_gather`
  - 去掉 padding，恢复完整 token 顺序

- `forward`
  - 如果阶段一未启用，直接回退原生 wrapper
  - 如果阶段一启用，只替换 `fused_qkv_a_proj` 计算
  - 后续 MLA / sparse attention 路径保持原逻辑

辅助配置位于：

```text
vllm_fl/ops/dsa_cp.py
```

主要开关：

```text
FL_ENABLE_DSA_CP=1
FL_DSA_CP_MODE=a_proj
```

## 7. 启动方式

阶段一推荐启动方式：

```bash
export FL_ENABLE_DSA_CP=1
export FL_DSA_CP_MODE=a_proj

vllm serve /model/DeepSeek-V3.2 \
  --served-model-name ds32-dsa-cp \
  --tensor-parallel-size 8 \
  --gpu-memory-utilization 0.85
```

成功启用时，日志中应出现类似：

```text
CUDA DSA-CP experimental wrapper enabled in a_proj mode
CUDA DSA-CP a_proj ACTIVE: prefix=model.layers.0.self_attn, tp_rank=0/8, ...
```

## 8. 验证建议

### 8.1 功能验证

使用短输入先验证服务可启动、可返回结果：

```bash
curl http://127.0.0.1:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "ds32-dsa-cp",
    "prompt": "hello",
    "max_tokens": 16
  }'
```

### 8.2 长上下文 benchmark

DeepSeek-V3.2 tokenizer 需要使用 `deepseek_v32` tokenizer mode，否则 random dataset 可能生成远超预期的 token 数。

```bash
vllm bench serve \
  --backend openai \
  --endpoint /v1/completions \
  --model ds32-dsa-cp \
  --tokenizer /model/DeepSeek-V3.2 \
  --tokenizer-mode deepseek_v32 \
  --dataset-name random \
  --random-input-len 32000 \
  --random-output-len 128 \
  --random-range-ratio 0 \
  --num-prompts 4
```

### 8.3 对比项

建议至少对比两组：

```text
Baseline: 不设置 FL_ENABLE_DSA_CP
Phase 1:  FL_ENABLE_DSA_CP=1, FL_DSA_CP_MODE=a_proj
```

观察：

- 服务是否稳定启动
- CUDA graph capture 是否正常完成
- benchmark 是否无长度误报
- 首 token / 吞吐是否有变化
- 是否出现 NCCL timeout

## 9. 当前收益判断

阶段一的收益预期有限。

原因是 `fused_qkv_a_proj` 只是一部分重复计算，而 CUDA 原生路径中更重的 `q_b_proj`、`kv_b_proj`、`o_proj` 已经是 TP sharded。阶段一额外引入了一次 token 维度 all-gather，节省的 GEMM 计算可能被通信成本抵消。

因此阶段一更适合作为：

- 功能可行性验证
- CUDA DSA-CP wrapper 接入验证
- 后续更深层 DSA-CP 的安全基线

不应把阶段一视为完整 DSA-CP 最终性能方案。

## 10. 已知边界

阶段一不处理以下内容：

- 不切分 sparse indexer 的 `wq_b`
- 不切分 sparse indexer 的 `wk_weights_proj`
- 不改变 `SparseAttnIndexer` 自定义 op
- 不改变 FlashMLA Sparse backend
- 不改变 KV cache / indexer cache 布局
- 不实现 Ascend `layer_sharding` 的等价 CUDA 版本

这些边界是有意保留的，目的是保证阶段一不破坏原生 vLLM sparse attention 语义和 CUDA graph 稳定性。

## 11. 后续阶段方向

如果继续推进完整 DSA-CP，真正有价值的方向是 sparse indexer / DSA 前处理：

```text
indexer.wq_b
indexer.wk_weights_proj
SparseAttnIndexer 输入与 cache 更新
```

但这部分不能简单通过 Python monkey-patch 实现生产级 CUDA graph 兼容。更合理的方向是：

- 在调度器层保证 token 数按 TP 对齐
- 在 CUDA graph capture 前统一 shape 与 collective 顺序
- 将 indexer projection + 通信 + sparse indexer 封装成 graph-safe 路径
- 必要时增加 C++/CUDA custom op 或修改 vLLM attention backend

阶段一为这些工作提供了一个最小可运行基线，但不是最终完整 DSA-CP。

## 12. 结论

阶段一 CUDA DSA-CP 的设计可以概括为：

```text
只对 fused_qkv_a_proj 做 token-parallel，
all-gather 后恢复原生 MLA 输入，
其余 sparse attention 路径完全保持 vLLM 原生行为。
```

它与普通推理的核心区别是：普通推理每个 TP rank 都对全部 token 执行 `fused_qkv_a_proj`，阶段一让每个 TP rank 只处理一段 token，再通过 all-gather 还原完整结果。

该方案功能风险低、CUDA graph 兼容性好，但收益有限。它是 CUDA DSA-CP 的第一阶段可交付版本，而不是完整 DSA-CP 性能方案。
