MiniMax-M3 覆盖率与性能对照
======================

验收目标为 GPU 关联 ATen 与独立 GPU kernel 的合计 Gems + 已确认 Triton 类型覆盖率至少 80%，同 case 输出吞吐不低于原配置的 95%，同时报告 TTFT、TPOT。所有 kernel 按精确符号去重，不删除未覆盖项。

首轮全量替换为 47/53（88.68%），但吞吐约 574 tok/s，相比原配置约 1000 tok/s 下降约 43%，因此不作为推荐配置。现启动器默认全部沿用原配置；混合方案必须显式选取。

当前结论（最终配置，2026-10-14）
--------------------------------

验收三个场景：64 并发、256 请求、输出 1k、`--no-enable-prefix-caching`，指标为 bench 的 Total token throughput。下表数据均在同一台 16 卡 PPU 上，先用该输入长度跑 64 请求预热，再测 256 请求（`profiles/inlen_debug_20261012/final_*.log`、`base_*.log`）。

| 场景 | 参考值 | 目标（≥95%） | 原配置（本机实测） | 最终配置 | 结论 |
|---|---:|---:|---:|---:|---|
| 1k/1k | 3320 | 3154 | 3054.6 | 3060.4 | 未达标，-3.0%；比原配置略高 |
| 16k/1k | 5928 | 5632 | 5709.5 | 5670.4 | 达标 |
| 4k/1k | — | — | 输出 1004 tok/s（stable 轮） | 输出 985.5 tok/s（总 4927.3） | 达标（-1.8%） |

合计覆盖率 52/62（83.87%）：ATen 17/19，独立 kernel 35/43（`profiles/minimax_m3_final_cov_20261014/gpu_coverage`，verification 通过，16 rank 均有 graph replay）。未覆盖 kernel 为 DeepGEMM grouped GEMM（2）、pccl 通信（4）、computeBlockInfo、flashinfer 采样。

最终配置相对 v3 的改动和原因：

- **Gems `index` 放回黑名单**：它使 vLLM 预估的每个 PIECEWISE CUDA graph 显存从 54 MiB 升到 98 MiB，51 个 graph 合计约 1 GiB，全部从 KV cache 扣除。改回后 KV cache 为 547,328 token，比原配置的 519,936 多 5.3%，16k 场景因此达标。代价是 ATen 覆盖少一项。
- **decode MoE（≤128 token）走 DeepGEMM 并关闭其 GEMV 分支**：DeepGEMM 在小 M 时自动选 GEMV，其中小 K 版本（gemm2，K=192，M≤47）结果错误，误差约 1，输出量级约 0.3，且每次运行位置不同（`tools/ppu/minimax_m3/check_deepgemm_gemv.py`）。原配置 `VLLM_FL_MOE=deepgemm` 在 decode batch 5–47 时也会走这段错误 kernel。关闭后回退到 grouped GEMM，每个 M 都正确，速度与 GEMV 相差几个百分点，比最优 Gems tile 快 1.2–1.3 倍。prefill 仍走 Gems，以保留 `fused_moe_kernel` 等覆盖。
- **split-K Triton**（`splitk.py`）：用于 router（FP32，3xTF32）和小 M 的 qkv，比原生快 10–40%。
- **Gemma norm 缓存 fp32 的 `1+weight`**：每个 decode step 少约 240 次 kernel launch。

1k/1k 未达标的原因（`profiles/warmprof_{baseline,hybrid}_20261014`，decode-only step 逐 kernel 对比）：decode step 的 GPU 忙碌率在两边都是 99%，决定 step 时间的是主 stream。主 stream 上原配置 33.41 ms/step，最终配置 33.60 ms/step。其中 routed MoE（DeepGEMM，约 17 ms）、allreduce（约 4.3 ms）和稀疏注意力（约 4.7 ms）两边完全相同，占比约 78%。shared expert 的 dense GEMM 在两条侧 stream 上与 MoE 并行，hybrid 把它从 3.8 ms 降到 0.8 ms，但不在关键路径上，所以不改变 step 时间。主 stream 上 hybrid 可控的 dense、norm、qknorm 合计约 3.6 ms；冷权重全量 sweep（`sweep_mainstream.py`）显示剩余空间约 0.2 ms/step，而达标需要减少约 1.7 ms/step。原配置在本机也只有 3054.6，参考值 3320 的测量条件与本机不同。

v3 的历史结论
------------

| 指标 | 原配置（无 profiler，stable1/2） | hybrid v3（无 profiler，stable1/2） | 变化 |
|---|---:|---:|---:|
| 输出吞吐 | 1000.9 / 1007.6 tok/s | 971.0 / 971.2 tok/s | -3.3% |
| 平均 TTFT | 8434.6 / 8434.5 ms | 8646.3 / 8816.7 ms | +3.5% |
| 平均 TPOT | 55.57 / 55.14 ms | 57.27 / 57.11 ms | +3.4% |
| 合计覆盖率 | 40/73（54.79%） | 52/64（81.25%） | ATen 18/19，kernel 34/45 |

性能数据目录：`profiles/minimax_m3_stable_baseline_20261011`、`profiles/minimax_m3_stable_hybrid_v3_20261012`；覆盖率目录：`profiles/minimax_m3_hybrid_v3_20261012/gpu_coverage`（verification 通过，16 rank 均有 graph replay）。两组 stable 轮次均为 128 请求、零失败，输入/输出 token 对账一致。

hybrid v2 的无 profiler 两轮为 972.5 / 868.1 tok/s，波动来自服务期间的 Triton 即时编译：Gems `index` 的线性化 kernel 把被索引维长度写入源码，`moe_align_block_size_stage1` 把输出长度和 tile 声明为 constexpr，chunked prefill 几乎每步都有新 token 数，每次编译使 16 个 TP rank 一起停顿数秒（v2 stable2 有 225 次 >1s 的 ITL）。`shape_stable.py` 把这些仅用于掩码/跨度的尺寸改为运行期参数，kernel 计算与配置不变；v3 从 warmup2 起 >1s ITL 为 0，stable 轮次期间无新 Triton 编译。stage1 的 kernel 符号因此变为 `_moe_align_stage1_kernel`，仍作为已确认 Triton kernel 计入，覆盖分子不变。

profile 轮（带 torch profiler，冷编译缓存）吞吐为 753 tok/s，尾部仍有 Gems pointwise（`_copy_kernel` 等）首次编译；这些 kernel 按 rank/对齐有限特化，缓存热后不再出现，原配置同样经过这些 Gems 路径。profile 轮仅用于覆盖率采集，性能验收采用无 profiler 的 stable 轮。

代码位置
--------

本目录是 PPU（T-Head）上 MiniMax-M3 的专属优化，全部由环境变量显式开启，未设置时与原配置一致。共享代码只通过 `vllm_fl.optimizations` 的两个入口调用：`register_model_optimizations()`（`ops/custom_ops.py`）和 `select_moe_experts()`（`ops/fused_moe/fused_moe_utils.py`），非 PPU 设备会跳过并告警。

| 文件 | 内容 | 开关 |
|---|---|---|
| `__init__.py` | 开关解析、注册、MoE experts 选择 | — |
| `moe.py` | `HybridM3Experts`：≤128 token（decode）用 DeepGEMM 并关闭其 GEMV 分支（小 K GEMV 结果错误），prefill 用固定 tile 的 Gems | `VLLM_FL_MOE=hybrid` |
| `linear.py` | BF16 linear：M≤128 按 M 分段选 split-K 或 Gems tile；FP32 router M≤512 用 split-K；更大回原生 | `VLLM_FL_M3_DENSE=hybrid` |
| `splitk.py` | 插件 Triton split-K skinny GEMM（两遍确定性归约，FP32 用 3xTF32），非 FlagGems | 随 `VLLM_FL_M3_DENSE=hybrid` |
| `norm.py` | Gemma RMSNorm / fused add RMSNorm 用 Gems kernel，fp32 的 `1+weight` 按权重缓存 | `VLLM_FL_M3_GEMMA_NORM=flaggems` |
| `qknorm_rope.py` | 把融合 Q/K norm+RoPE+KV insert 路由到 Triton（kernel 在 `ops/minimax_m3_triton.py`） | `VLLM_FL_M3_FUSED_IMPL=triton` |
| `../common/shape_stable.py` | Gems `index` / `moe_align` stage1 的运行期尺寸版本，避免服务中重编译（默认 index 走原生，见下） | 任一开关开启即安装 |

测试在 `tests/optimizations/ppu/minimax_m3/`，启动/压测/覆盖率脚本在 `tools/ppu/minimax_m3/`。新模型按 `optimizations/<vendor>/<model>/` 新建目录（提供 `VENDOR`、`enabled()`、`register()`，可选 `moe_experts()`），并加入 `optimizations/<vendor>/__init__.py` 的 `_MODELS`。

启动器固定保存启动参数、相关环境、修改文件哈希和 benchmark case：

```bash
python3 tools/ppu/minimax_m3/serve_minimax_m3_gems.py \
  --model /mnt/cpfs/xiangbin/MiniMax-M3 \
  --run-dir /mnt/cpfs/xiangbin/codex/profiles/minimax_m3_gems_NEW_RUN \
  --moe hybrid --dense hybrid --norm flaggems --fused triton --index native \
  --profile --log-file
```

默认使用 16 卡、8100 端口、显存利用率 0.85、32768 最大长度、block-size=128；run-dir 必须是新目录。运行依赖当前环境已安装的 FlagGems、vLLM 与覆盖率 skill。该工作区链接已有 `_m3_fused_C.so` 作为原生 schema 和对照参考，没有重新编译原生扩展。

等 `/health` 正常后，执行相同 case 的两轮预热与一轮 profile：

```bash
python3 /root/.codex/skills/vllm-gpu-coverage/scripts/run_benchmark.py \
  --command-json RUN/bench.json --out RUN/bench \
  --base-url http://127.0.0.1:8100 --flaggems-log RUN/flaggems.txt \
  --expected-prompts 128 --expected-input-tokens 524288 \
  --expected-output-tokens 131072
```

以上混合候选使用以下选择（启动器无选项时为原配置）：

```bash
VLLM_FL_TOPK_DISPATCH=1
VLLM_FL_MOE=hybrid
VLLM_FL_FLAGOS_BLACKLIST=index_copy_,index,linear,addmm,mm,empty.memory_format
VLLM_FL_M3_DENSE=hybrid
VLLM_FL_M3_GEMMA_NORM=flaggems
VLLM_FL_M3_FUSED_IMPL=triton
```

- `VLLM_FL_M3_GEMMA_NORM=flaggems` 接管模型直接调用的 Gemma RMSNorm。普通分支使用 FlagGems 的 Gemma kernel；残差分支使用 FlagGems 的 add 和单遍 fused-add RMSNorm，保持 FP32 的 `1 + weight`、累加与两个 tensor 原地写回。接口用 custom-op schema 和 fake implementation 保持编译/graph 的别名语义。
- 显式 `VLLM_FL_M3_FUSED_IMPL=triton` 在原生 schema 存在时也覆盖 CUDA dispatcher 的实现，原生扩展不再使此选择失效。融合 kernel 当前属于插件中的 Triton 实现，不能称为已上游至 FlagGems。当前支持 auto BF16/FP16 KV cache；其他 cache dtype 明确拒绝。修正了没有 KV 插入时稀疏 IQ/IK 的 packed head 映射，以及 dense cache 插入不应该强制访问 index_cache 的检查。
- `VLLM_FL_MOE=flaggems` 使用已有 TritonExpertsFL 模块化路径。修正旧快速入口对 SiLU 以外的激活与 clamp 配置误选；MiniMax 保留 SwiGLU-OAI 的 alpha=1.702、beta=1、clamp=7。GEMM 和调度可复用已有 FlagGems 实现。
- 混合 MoE 仅在 64–128 token 保留 DeepGEMM，其余 batch 用调参后的现有 Gems 实现。针对 BF16、E=128、topk=4、K=6144、I=192。独立参考发现原生 small-K GEMV 在 31/32 token 有不稳定元素，因此 sub-64 全用 Gems，误差阈值没有放宽。
- `VLLM_FL_M3_DENSE=hybrid` 接入 BF16 非量化 linear：decode 使用微基准选择的 tile，vocab 复用现有 Gems 选择；M>128 的大批量 dense prefill 恢复原生，微基准中 Gems 慢 15–40%。FP32 路由与 mm/addmm 保留原生；必须保留 `linear,addmm,mm` 黑名单，防止 fallback 再次被全局替换。
- `index` 默认保留在黑名单：Gems index 使每个 PIECEWISE CUDA graph 的预估显存从 54 MiB 升到 98 MiB，51 个 graph 合计约 1 GiB，KV cache 从 537,728 降到 497,152 token（原配置 519,936），16k 输入时直接降低并发。代价是 ATen 覆盖少 1 项（18/19→17/19）。
- `--index flaggems` 解除 index 黑名单，保留 index_copy_ 与 empty.memory_format。已有整数/负索引和 graph 测试；整机采集进一步验证实际用法。系统安装的原插件未修改，工作区通过 PYTHONPATH 加载。
- 开启 Gems index 时启动器设置 `FLAGGEMS_DB_URL=sqlite:///:memory:`，让每个 worker 独立缓存。当前 Gems 的 policy 子类会覆盖 LibTuner 基类方法，禁用运行期调优必须同时覆盖注册子类。hybrid v1 因动态 index 在服务中调优、共享 SQLite 锁冲突而退出，输出 token 对账失败，没有完整 profile；该轮不能参与验收。

可自动管理服务、压测和停止流程：

```bash
python3 tools/ppu/minimax_m3/run_minimax_experiment.py --run-dir /mnt/cpfs/xiangbin/codex/profiles/minimax_hybrid_NEW
python3 tools/ppu/minimax_m3/analyze_minimax_run.py /mnt/cpfs/xiangbin/codex/profiles/minimax_hybrid_NEW
```

`benchmark_minimax_backends.py` 测量空闲 GPU 上的 graph 延迟，不能替代整机压测。`check_minimax_moe_reference.py` 用独立公式定位小 batch 原生实现差异。

验证命令：

```bash
python3 -m pytest tests/optimizations/ppu/minimax_m3 -q
```

`test_shape_stable.py`（7 项）对照 torch 索引与独立 moe_align 参考，覆盖多种行数、负索引、前缀维、多索引回退和 graph replay，并断言改变行数/token 数不再新增编译。该目录全部 72 项通过；装入补丁后 MoE 与 hybrid 的 38 项也重跑通过。

首轮结果为 15、3、12 项通过。新增混合 BF16 linear/index 的 19 项测试通过（M=1/64/129/8192、偏置、graph）；扩展后的 MoE 16 项通过，M=1/31/32/63/64/128/129/257，每个 token 对照独立 FP32 公式和 BF16 中间边界，并检查 graph replay。BF16 width=6144、E=128/K=6144/I=192/topk=4；保留 SwiGLU-OAI 参数与路由权重。测试形状不能替代完整服务 profile 或生成质量评估。

完整采集需解析所有 16 个 worker trace、确认实际 graph kernel 事件、收集日志与编译 metadata，再按 skill 口径导出报告。kernel 数量可能因实现变化而变化，不能直接把旧未覆盖项数量加到旧分子，或将消失的原生 kernel 计为已覆盖。通信和采样仍由原库提供，覆盖结果以实际 trace 为准。
