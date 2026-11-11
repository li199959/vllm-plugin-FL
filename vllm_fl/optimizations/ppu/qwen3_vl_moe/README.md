Qwen3-VL-30B-A3B 覆盖率与性能
==========================

目标：GPU 关联 ATen 与独立 GPU kernel 合计覆盖率 ≥80%（vllm-gpu-coverage skill 口径）；三个场景的输出吞吐 ≥ 参考值的 95%。

三个场景均为 64 并发、128 请求、输出 1k、ignore-eos，全部加 `--no-enable-prefix-caching`。启动命令与用户一致：`-tp 2 --max-model-len 8192 --gpu-memory-utilization 0.9 --max-num-seqs 128`。16k 输入超过 8192，16k 场景改用 `--max-model-len 20480`。

> **2026-10-20 更正**：此前的版本没有关闭 prefix cache。bench 没有 `--seed`，每轮发送相同的 prompt，第 2 轮起 prefill 大多命中 cache，1k/4k 的吞吐因此偏高，那些数据作废。本文所有数据都在关闭 prefix cache 后重新测量。

结论
----

三个场景使用同一套配置（v11），只有 `--max-model-len` 不同。

```bash
CUDA_VISIBLE_DEVICES=0,1 \
PYTHONPATH=/mnt/cpfs/xiangbin/codex/minimax-m3-coverage-80 \
VLLM_DISABLE_COMPILE_CACHE=1 \
VLLM_USE_FLASHINFER_SAMPLER=0 \
VLLM_FL_FLAGOS_BLACKLIST=mm,mm_out,addmm,addmm_out,linear,empty \
VLLM_FL_QWEN3VL_STABLE=1 \
VLLM_FL_QWEN3VL_ATTN=fa3 \
VLLM_FL_QWEN3VL_MOE=hybrid \
VLLM_FL_QWEN3VL_DENSE=tuned \
vllm serve /mnt/cpfs/xiangbin/Qwen3-VL-30B-A3B-Instruct/ \
    --served-model-name qwen3-vl -tp 2 --port 8000 \
    --max-model-len 8192 --gpu-memory-utilization 0.9 --max-num-seqs 128 \
    --no-enable-prefix-caching          # 16k: --max-model-len 20480
```

| 场景 | 参考值 | 目标（≥95%） | v11（第 1–3 轮） | 平均 TTFT（参考） | 平均 TPOT（参考） |
|---|---:|---:|---:|---:|---:|
| 1k/1k | 2523.18 | 2397 | 2443 / 2441 / 2454 | 1465 ms（1357） | 24.7 ms（24.04） |
| 4k/1k | 1575.70 | 1497 | 1503 / 1509 / 1529 | 4322 ms（4088） | 38.0 ms（36.50） |
| 16k/1k | 491.09 | 466.5 | 491 / 492 | 22185 ms（22285） | 107.7 ms（107.81） |

上表各轮 128/128 完成，无失败。目录：`profiles/q_v11_{1k,4k,16k}_20261019`。

- 1k 单独压测。4k 和 16k 是两个服务同时压测的。
- 16k 跑在 NV2 卡对（GPU 12/13）上。本机 GPU 0/1、4/5、8/9、12/13 是 NV2 卡对，2/3、6/7、10/11、14/15 是 NV1 卡对；16k 在 NV1 卡对上约慢 4%。
- 4k 余量最小，只有 0.4–2%。

**覆盖率**：v11 合计 53/65（81.54%），其中 ATen 23/24、独立 kernel 30/41。verification 通过，见 `profiles/q_v11_cov_20261020/gpu_coverage`。未覆盖的 12 项：

| 未覆盖项 | 数量 | 说明 |
|---|---:|---|
| `aten.mm` | 1 | 走 cuBLAS |
| DeepGEMM grouped GEMM | 2 | |
| DeepGEMM permute `computeBlockInfoKernel` | 1 | |
| FA3 kernel | 5 | 3 个 forward、1 个 combine、`prepare_varlen` |
| pccl 通信 | 3 | |

原配置的覆盖率是 39/44（88.64%），但这台机器上关闭 prefix cache 后，原配置 4k 只有 793 tok/s。

**精度**：v11 与原配置使用同一台机器、同一套评测方法，对比如下。

| 数据集 | 原配置 | v11 |
|---|---:|---:|
| GPQA Diamond（198 题） | 54.55% | 57.58% |
| ERQA（400 题） | 31.25% | 31.25% |

- 评测方法按 `profiles/qwen36_35b_a3b_eval_20261007/REPORT.md`：EvalScope 1.12，同一份本地数据集，rule 评分，batch 32，seed 42。
- 本模型是 Instruct 模型，所以采样参数采用其 generation_config（T=0.7、top_p=0.8、top_k=20），不开 thinking。
- 服务端 `--max-model-len 32768`，`max_tokens` 为 16384。
- 脚本：`tools/ppu/qwen3_vl_moe/eval_qwen3_vl.sh`。结果在 `profiles/q_eval_{v11,native}_{gpqa,erqa}_20261020`。
- 两种配置在 ERQA 上的答案首字母一致率为 309/346。
- 加上官方采样参数（presence_penalty 1.5，`max_tokens` 32768）：v11 57.07%（3 次，594 样本），原配置 59.60%（1 次，198 样本），差距在统计误差内。

**GPQA 与官方 70.4 的差距来自 prompt**。官方值见 Qwen3-VL 技术报告（arXiv 2511.21631）Table 4，采样参数与上一条相同。`gpqa_prompt.py` 在 v11 上按官方参数对照了几种 prompt（198 题 × 3 次，选项按固定种子打乱）：

| Prompt | 准确率 | 输出中位数（token） |
|---|---:|---:|
| boxed："Please reason step by step, and put your final answer (the option letter) within \boxed{}." | **71.55%** | 3847 |
| Qwen 推荐的 JSON `answer` 字段 | 68.18% | 2697 |
| EvalScope 默认（"ANSWER: [LETTER]"） | 60.44% | 496 |
| EvalScope 默认 + system prompt | 58.59% | 504 |
| simple-evals（"Answer: $LETTER"） | 57.41% | 496 |
| EvalScope 默认 + 贪心解码 | 55.89% | 498（34 条重复截断） |

末行格式类的模板会把推理压到约 500 token；boxed / JSON 写法下模型完整推理，分数与官方一致。对外报 GPQA 用 boxed prompt。boxed 只在 v11 上测过，原配置未测。结果在 `profiles/q_prompt_eval_20261020`。

**已知问题**：ERQA 中有少量输出以一串 `!` 开头（token id 0），后面是乱码。

| 配置 | 出现条数（/400） |
|---|---:|
| 原配置 | 17 |
| v11 | 6 |
| 去掉 STABLE | 0 |
| MOE=tuned | 35 |

这个问题原配置本身就有，不是 v11 引入的。单独调用 MoE kernel 测试时，M=1500/4100/8192 以及偏斜路由下都没有出现 NaN，根因尚未定位。

各开关的作用
------------

| 开关 | 作用 | 依据 |
|---|---|---|
| `STABLE=1`（`../common/shape_stable.py`） | Gems index、moe_align stage1、copy_ 改为运行期传入尺寸 | 原配置每个 token 数都会重编译（Gems pointwise 把 stride 当作 constexpr；M-RoPE 的 [3,N] 切片 copy），服务期间 ITL 停顿 9–29 s |
| `ATTN=fa3`（`attention.py`） | 所有合格的 step 都走 PPU FA3 wheel（`torch.ops.flash_attn_3.fwd`）；含 prefill 的 step 用 `num_splits=1`（不增加耗时，少一个 Combine 变体） | prefill 比 Triton 2D 快 3.4–4.2x；真实服务中 decode 快约 13% |
| `MOE=hybrid`（`moe.py`） | 恰好 8192 token 的完整 prefill chunk 走 DeepGEMM；其余 M 走调过 tile 的 Gems（PPU 上没有内置 tile 表） | 8192 时 DeepGEMM 快 1.84x。阈值可用 `VLLM_FL_QWEN3VL_DEEPGEMM_MIN` 调整；改到 1024 会多出 2 个未覆盖的 DeepGEMM 变体 |
| `DENSE=tuned`（`linear.py`） | decode（M≤128）时 qkv、o_proj、router 走 Gems `linear_kernel` 的按 M tile | 比 Gems mm 默认 tile 快 1.3–2.8x |
| 黑名单 `mm_out,addmm,addmm_out,linear` | prefill 的大矩阵乘改走 cuBLAS | Inductor 调用的是 `mm.out`；黑名单按函数名匹配，只写 `mm` 拦不住 |
| 黑名单 `mm` | lm_head 等直接调用 `aten.mm` 的地方也走 cuBLAS | 去掉这一项覆盖率是 83.08%，但 4k/16k 会慢 2–3% |
| 黑名单 `empty` | `torch.empty` 不再启动 kernel | Gems 的 empty 每次调用都启动一个真实 kernel，去掉后 1k 快约 9%，4k 快约 6% |
| `VLLM_DISABLE_COMPILE_CACHE=1` | 每次启动重新编译 | vLLM 编译缓存的 key 不包含 `VLLM_FL_*`，切换开关后会复用旧图，DENSE 等开关不会生效 |
| `VLLM_USE_FLASHINFER_SAMPLER=0` | top-k/top-p 改走 vLLM 的 Triton 实现，采样参数不变 | 去掉 2 个未覆盖的 flashinfer kernel，ATen 覆盖增加 10 余项，吞吐无可测差异 |

测量中发现的问题：

- **prefix cache**：吞吐对比必须关闭 prefix cache，否则第 2 轮起 TTFT 只有参考值的约十分之一。
- **NV1/NV2 卡对**：16k 在 NV1 卡对上约慢 4%。
- **CPU 争用**：多个服务同时压测时机器负载 >20，1k 会降 2–4%。

代码位置
--------

| 文件 | 内容 |
|---|---|
| `__init__.py` | 开关解析、注册、`moe_experts()` |
| `attention.py` | 3D decode launch 参数；FA3 dispatch（`hybrid` 只走 prefill，`fa3` 全部走） |
| `moe.py` | Gems MoE tile 表；`HybridQwen3VLExperts` |
| `linear.py` | decode 尺寸的 qkv、o_proj、router |
| `../common/shape_stable.py` | 运行期尺寸的 Gems index、moe_align stage1、copy_，与 M3 共用 |

共享代码只通过 `register_model_optimizations()` 和 `select_moe_experts()` 两个入口调用。系统安装的插件未修改。

工具（`tools/ppu/qwen3_vl_moe/`）：

| 用途 | 脚本 |
|---|---|
| 启动与压测 | `serve_qwen3_vl.sh`（`MAX_LEN`、`EXTRA_ARGS`）、`bench_qwen3_vl.sh`（`INPUT_LEN`） |
| 覆盖率 | `coverage_qwen3_vl.sh` |
| 精度评测 | `eval_qwen3_vl.sh`（EvalScope）、`gpqa_prompt.py`（GPQA prompt 对照，官方采样参数） |
| profile | `warm_profile.sh` |
| 微基准 | `sweep_*.py`（tile 表与 launch 参数的来源）、`bench_*.py`（FA3 / DeepGEMM 的选择依据） |

测试：`python3 -m pytest tests/optimizations -q`，193 项全部通过，含 M3 的测试。
