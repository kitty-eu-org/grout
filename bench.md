# grout prefill 性能对比基准

针对 [cutile-rs issue #171](https://github.com/NVlabs/cutile-rs/issues/171#issuecomment-4781966468)(grout prefill 在 RTX 5070 Ti 上明显慢于论文 artifact),本文档记录了 grout(Rust)与 PyTorch transformers 的 **prefill 专用**性能对比,并定位了瓶颈所在。

> **只对比 prefill。** issue #171 的核心是 prefill 慢,decode 不在范围内。grout 跑 `--max-new-tokens 0`,transformers 跑单次 forward。

---

## 环境与配置

| 项目 | 值 |
|---|---|
| GPU | NVIDIA GeForce RTX 5070 Ti(Blackwell sm_120,16 GB) |
| 模型 | Qwen3-4B-Instruct-2507(`Qwen3ForCausalLM`,36 层,`tie_word_embeddings=true`) |
| dtype | `float16`(grout 把 bf16 权重在加载时转 f16;transformers 用 `.half()`) |
| attention | grout: flash-attn kernel / transformers: `scaled_dot_product_attention`(eager,**未** `torch.compile`) |
| prefill | 两边都是**单次 batched forward**,无 chunking |
| chat template | 两边用**同一个字面 prompt**:`<\|im_start\|>user\n{P}<\|im_end\|>\n<\|im_start\|>assistant\n<think>\n\n</think>\n\n`(grout `maybe_apply_chat_template`,`src/model.rs:1264`) |
| 采样 | 关闭(greedy argmax) |
| warmup | 3 次,不计入测量 |
| 计时 | grout: Rust `Instant`,含隐式 GPU 同步 / transformers: CUDA event `t0.elapsed_time(t1)`,纯 GPU 时间 |

prompt token 数两边一致(541),`compare.py` 会校验。

### grout 优化开关
```
GROUT_CUBLAS_COMPUTE16=1   # FP16 累加
GROUT_PROFILE_OPS=1        # 输出 per-op 计时
```

---

## 实测结果(541 prompt tokens)

| 指标 | grout (Rust) | transformers (PyTorch) |
|---|---|---|
| **prefill tok/s** | **2,120** | **8,096** |
| **prefill ms** | **255** | **67** |
| prompt tokens | 541 | 541 |

**→ grout prefill 慢约 3.82×。** 这是对比**未优化**的 PyTorch eager,所以相对论文 artifact 差距只会更大。

### 各 run 的一致性(grout,warmup 后)
```
run 1: 254.69 ms
run 2: 254.57 ms
run 3: 255.39 ms
run 4: 255.57 ms
run 5: 255.78 ms
```
方差极小,说明是**稳定可复现的 GPU 执行瓶颈**,不是随机波动。

---

## 瓶颈定位

### 关键发现:98% 的时间 host timer 看不见

用 `GROUT_PROFILE_OPS=1` 抓到的 per-op **host launch 时间**(CPU dispatch 间隔,不含 GPU 执行):

```
grout prefill (last run):  255.78 ms
sum of op launches:        5.80 ms (2% of prefill)
UNACCOUNTED GPU time:      249.98 ms (98% of prefill)   ← 真实瓶颈在这

  op                launch_ms   share   calls   avg_us
  ----------------------------------------------------
  MatMul               2.233    0.9%     252     8.86
  RmsNorm              1.219    0.5%     145     8.41
  Rope                 0.587    0.2%      72     8.15
  Add                  0.577    0.2%      72     8.01
  Attention            0.378    0.1%      36    10.50
  KvCacheUpdate        0.351    0.1%      36     9.75
  SiluMul              0.272    0.1%      36     7.56
  ...
```

### 为什么 op_profile 只是 launch 时间

核对了 grout 源码 `src/model.rs:1743-1906`:

```rust
let op_start = if profile_ops { Some(Instant::now()) } else { None };  // 1744
// ... 执行 op(kernel 异步 launch)...
if let Some(op_start) = op_start {
    self.profile_op(graph_op_name(op), op_start.elapsed());            // 1906
}
```

中间**没有 `cuda.synchronize`**,所以 `op_start.elapsed()` 只量到 CPU 把 kernel 派发出去的间隔(~8 µs/op),**不包含 kernel 在 GPU 上实际跑的时间**。因此总和只有 ~6ms,而真实 prefill 是 255ms。

### 诊断结论

**慢在 kernel 执行本身**,不是 launch overhead,也不是 JIT 编译:

- **不是 launch overhead** —— 252 次 MatMul launch 总共才 2.2ms,与 255ms 差两个数量级。
- **不是 JIT 编译** —— warmup 第一跑 745ms(含首次编译),但 warmup 后稳态仍是 257ms;测量的是稳态。
- **是真实 GPU kernel 执行慢** —— 那 250ms 是 kernel 在 GPU 上排队/执行的真实耗时。

最大嫌疑(按 FLOPs 占比):
1. **cuBLAS GEMM**(MatMul):q/k/v/o + gate/up projections,36 层 × 7 = 252 次,是 prefill 算力主体。
2. **flash-attn prefill kernel**。

### 下一步:精确到每个 kernel

op_profile 给不出真实 GPU 时间。要精确定位,用 nsys/ncu:

```bash
nsys profile -t cuda ./target/release/grout \
    --model <Qwen3-4B> --prompt "<长prompt>" \
    --max-new-tokens 0 --raw-prompt
```

---

## 复现

### 一键脚本

```bash
# 用 uv 自建 torch 环境(会下载 cu128 wheel):
bench/run_comparison.sh

# 或复用已有 torch venv(更快):
PY=/path/to/.venv/bin/python bench/run_comparison.sh
```

可覆盖默认值:`PROMPT="..." WARMUP=3 RUNS=5 bench/run_comparison.sh`

### 手动分开跑

```bash
# grout — prefill only,优化全开,带 profiling
#   PATH 必须含 tileiras + ptxas(cutile JIT 运行时需要)
CUDA_TOOLKIT_PATH=/usr/local/cuda-13 PATH="/usr/local/cuda-13/bin:$PATH" \
GROUT_CUBLAS_COMPUTE16=1 GROUT_PROFILE_OPS=1 \
cargo run --release -- \
    --model /path/to/Qwen3-4B-Instruct-2507 \
    --prompt "<长prompt>" \
    --max-new-tokens 0 --warmup-runs 3 --runs 5 \
    --raw-prompt --profile \
    --json-out bench/out/out_grout.json

# transformers — prefill only(单次 forward)
python bench/bench_transformers.py \
    --model /path/to/Qwen3-4B-Instruct-2507 \
    --prompt "<长prompt>" \
    --warmup-runs 3 --runs 5 \
    --raw-prompt \
    --json-out bench/out/out_transformers.json

# 对比 + 瓶颈分析
python bench/compare.py \
    --grout bench/out/out_grout.json \
    --transformers bench/out/out_transformers.json
```

---

## 工具说明

| 文件 | 作用 |
|---|---|
| `src/main.rs`(`--json-out`) | grout 把指标 + op-profile 写成 JSON(与 Python 同 schema);**不改动任何推理/计时逻辑** |
| `bench/bench_transformers.py` | transformers prefill 基线,单次 forward,CUDA event 计时 |
| `bench/schema.py` | 共享 JSON 契约(镜像 grout `GenerationOutput` 的指标定义) |
| `bench/compare.py` | 并排对比表 + 瓶颈定位(op_profile 解析 + unaccounted GPU time 诊断) |
| `bench/run_comparison.sh` | 一键编排:构建 → 跑两边 → 对比 |
| `bench/pyproject.toml` | uv 环境:torch(cu128 for Blackwell sm_120)、transformers |

### grout 可调环境变量

| 变量 | 作用 |
|---|---|
| `GROUT_CUBLAS_COMPUTE16=1` | cuBLAS 用 FP16 累加(更快,精度略降) |
| `GROUT_CUBLAS_COMPUTE16_MAX_M=<n>` | 限制 FP16 compute 的 M 维上限 |
| `GROUT_CUBLAS_FAST_ALGO=1` | 让 cuBLAS 自选快速算法 |
| `GROUT_ATTN_BN_PREFILL=<16\|32\|64>` | prefill attention 的 block-N tiling |
| `GROUT_PROFILE_OPS=1` | 输出 per-op host-launch 计时 |
