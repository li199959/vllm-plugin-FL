bge-m3 覆盖率与性能
==================

> **2026-10-22 更新**：最终配置为 v9，见 `REPORT_20261022.md`。下文的 v7 以“只有 attention 走 Gems 的插件配置”为基线，而该基线用 Triton attention，比用户提供的原生 vLLM 慢一倍以上；以原生为基线时 v7 不达标。v9 在 v7 上加 `VLLM_FL_BGEM3_ATTN=fa3`、`VLLM_FL_BGEM3_FA3_MIN_TOKENS=2048`，并把 `GEMS_SMALL_M` 提到 4096。

模型：BAAI bge-m3（XLMRobertaModel，24 层，hidden 1024，fp16），只用 dense 输出（CLS + normalize），TP1，`vllm serve --runner pooling`，走 `/v1/embeddings`。

目标：GPU 关联 ATen 与独立 GPU kernel 合计覆盖率 ≥80%（vllm-gpu-coverage skill 口径）；吞吐 ≥ 参考配置的 95%；embedding 与 fp32 参考一致。

结论
----

```bash
CUDA_VISIBLE_DEVICES=<gpu> \
PYTHONPATH=/mnt/cpfs/xiangbin/codex/minimax-m3-coverage-80 \
VLLM_DISABLE_COMPILE_CACHE=1 \
VLLM_FL_FLAGOS_BLACKLIST_APPEND=broadcast_to,mm,mm_out,addmm,addmm_out,linear,empty \
VLLM_FL_BGEM3_STABLE=1 \
VLLM_FL_BGEM3_DENSE=hybrid \
VLLM_FL_BGEM3_GEMS_LINEARS= \
VLLM_FL_BGEM3_GEMS_SMALL_M=512 \
vllm serve /mnt/cpfs/xiangbin/codex/bge-m3 --served-model-name bge-m3 --runner pooling \
  --max-model-len 8192 --gpu-memory-utilization 0.9 \
  --compilation-config '{"max_cudagraph_capture_size":4096}'
```

| 场景 | 参考 | v7（上面的配置） | 比值 |
|---|---:|---:|---:|
| 512 输入，64 并发，2048 请求 | 240.2 req/s | 237.7 req/s | 99.0% |
| 8192 输入，64 并发，256 请求 | 28.1k tok/s | 28.0k tok/s | 99.7% |
| 512 输入，4 并发，512 请求 | 161.8 req/s，24.2 ms | 146.4 req/s，26.7 ms | 90.5% |

- 覆盖率 15/17 = **88.24%**（ATen 10/11，kernel 5/6），未覆盖：`aten.addmm`、一个 cuBLAS gemm tile。
- 精度：12 条文本（中、英、德、日，含约 400 词和约 3000 词长文）与 fp32 HF 参考（CLS + normalize）的余弦最低 0.99998，5 条 query 的检索排序与参考一致。
- 参考配置：同一插件，只有 attention 走 Gems（`VLLM_FL_FLAGOS_WHITELIST=attention_backend`），其余为 torch/Inductor/cuBLAS。不带插件的纯 vLLM 起不来（thead attention 在 vLLM 0.24 上 import 失败）。两边的 `max_cudagraph_capture_size` 一致。
- 4 并发差 9.5%：即使所有 GEMM 都走 cuBLAS（v8）也是 147 req/s，差距来自 graph 外 eager 执行的小 Gems 算子（pooler 的 index/norm/div、输入准备）的 CPU 分发开销，不来自 GEMM。

原始配置的问题
--------------

1. **启动崩溃**。XLM-R 的 linear 带 bias，走 Gems addmm → `broadcast_to`；Gems 的 `broadcast_to` 做 pinned H2D 拷贝，CUDA graph 捕获时不允许（`operation not permitted when stream is capturing`）。Qwen/M3 的 linear 没 bias，不触发。修复：黑名单 `broadcast_to`，aten 的实现是零拷贝 view。
2. **第一个请求就把 engine 打挂**。插件 `worker/worker.py` 的 `execute_model` 只放行 `ModelRunnerOutput | None`，pooling 模型异步调度返回 `AsyncModelRunnerOutput`，落到 PP 分支的 assert。已按上游 `gpu_worker` 补上这个类型（worktree 内修改）。
3. **每个 batch 都卡顿**（p99 最高 14 s）。CLS pooler 用 `aten.index` 从 `[num_tokens, 1024]` 取行，Gems 把 num_tokens 写进生成的 index kernel，几乎每个 batch 都重新生成并 autotune。`VLLM_FL_BGEM3_STABLE=1` 复用 `common/shape_stable.py`。Gems 生成的源码在 `~/.flaggems/code_cache`，只看 `~/.triton/cache` 会漏数（bench 脚本已同时统计两处）。
4. **GEMM 慢**。M=8192 时 Gems mm/addmm 占 GPU 时间 77%，单测比 cuBLAS 慢 1.2–1.6 倍；全走 Gems 时 512 输入只有 153 req/s（-36%）。
5. **Gems `empty` 的 CPU 开销**。每层 `unified_kv_cache_update` 两次 `aten::empty`，Gems 实现每次约 60 µs，低并发时很明显（拉黑后 4 并发 76 → 94 req/s）。

开关
----

| 开关 | 作用 |
|---|---|
| `VLLM_FL_BGEM3_STABLE=1` | shape-stable Gems index / copy_ |
| `VLLM_FL_BGEM3_DENSE=hybrid` | 接管 `UnquantizedLinearMethod.apply`，按下面两个变量决定哪些 linear 走 Gems `linear_kernel`（按 M 选 tile），其余调 `F.linear`（mm 已拉黑，即 cuBLAS） |
| `VLLM_FL_BGEM3_GEMS_LINEARS` | 任意 batch 都走 Gems 的投影，`qkv,o,fc1,fc2` 的子集，默认 `o`，最终配置为空 |
| `VLLM_FL_BGEM3_GEMS_SMALL_M` | token 数 ≤ 该值的 batch 所有投影都走 Gems，默认 0（关） |

`GEMS_SMALL_M=512` 的作用：≤512 的 batch 是 CUDA graph 尺寸和尾 batch，GPU 时间占比小，但否则会各自带出 cuBLAS 小 tile 和 split-K reduction kernel。M 在编译图里是符号，所以大小分流在 custom op 内部运行时判断。

版本对比
--------

512 输入 64 并发，覆盖率按 skill 口径。同一时段、独占端口测量（之前有一批数据因两个服务绑同一端口而混测，已作废）。

| 版本 | 配置 | 64 并发 | 4 并发 | 覆盖率 |
|---|---|---:|---:|---:|
| 参考 | 只 attention 走 Gems | 238–241 | 158–166 | – |
| v3 | broadcast_to + STABLE，GEMM 全走 Gems | 153 | – | 17/17 |
| v4 | + mm/addmm/linear 拉黑 | 235 | 114 | 15/20 |
| ho | v4 + o_proj 走 Gems | 229 | 94 | 17/22 |
| s512 | ho + SMALL_M=512 | 232 | 76 | 16/17 |
| v5 | s512 + empty 拉黑 | 235 | 94 | 15/16 |
| v6 | v5 + capture 4096 | 234 | 142 | – |
| **v7** | v6，o_proj 不再固定走 Gems | **238** | **146** | **15/17** |
| v8 | v7 去掉所有 Gems linear | 236 | 147 | – |

目录
----

- 代码：本目录（`__init__.py` 开关，`linear.py` 混合 linear）；`vllm_fl/worker/worker.py` 的 AsyncModelRunnerOutput 修复。
- 测试：`tests/optimizations/ppu/bge_m3/`（GPU 正确性、CUDA graph replay、开关）。
- 脚本：`tools/ppu/bge_m3/`
  - `serve_bge_m3.sh <run_dir> <gpu> <port> <native|worktree> [profile]`（`EXTRA_ARGS`、`GPU_UTIL`）
  - `bench_bge_m3.sh <run_dir> <port> [rounds]`（`INPUT_LEN`、`CONC`、`NUM`，每轮不同 seed）
  - `coverage_bge_m3.sh <run_dir> <port>`：skill 的 `run_benchmark.py` 要求结果里有 `failed`，embedding 结果没有，所以三轮由脚本自己驱动
  - `check_embeddings.py`：fp32 参考与对比
  - `sweep_linear.py`：四种投影的 tile 扫描
- 数据：`/mnt/cpfs/xiangbin/codex/profiles/bge_m3_v7_2014`、`bge_m3_v7_cov_2014`（覆盖率报告在 `gpu_coverage/`）、`bge_m3_torchref_g4k_2008`、`bge_m3_torchref_1950`、`bge_m3_ref/`。
