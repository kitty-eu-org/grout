# grout prefill performance benchmark

This document records a **prefill-only** performance comparison between grout
(Rust) and PyTorch transformers for [cutile-rs issue #171](https://github.com/NVlabs/cutile-rs/issues/171#issuecomment-4781966468)
(grout prefill is much slower than the paper artifact on RTX 5070 Ti), and
locates the bottleneck.

> **Prefill only.** Issue #171 is about prefill; decode is out of scope. grout
> runs with `--max-new-tokens 0`; transformers runs a single forward.

---

## Environment & configuration

| Item | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 5070 Ti (Blackwell sm_120, 16 GB) |
| Model | Qwen3-4B-Instruct-2507 (`Qwen3ForCausalLM`, 36 layers, `tie_word_embeddings=true`) |
| dtype | `float16` (grout casts bf16 weights to f16 at load; transformers uses `.half()`) |
| attention | grout: flash-attn kernel / transformers: `scaled_dot_product_attention` (eager, **no** `torch.compile`) |
| prefill | both sides run a **single batched forward**, no chunking |
| chat template | both sides use the **same literal prompt**: `<\|im_start\|>user\n{P}<\|im_end\|>\n<\|im_start\|>assistant\n<think>\n\n</think>\n\n` (grout `maybe_apply_chat_template`, `src/model.rs:1264`) |
| sampling | off (greedy argmax) |
| warmup | 3 runs, not counted |
| timing | grout: Rust `Instant`, with implicit GPU sync / transformers: CUDA event `t0.elapsed_time(t1)`, pure GPU time |

Prompt token counts are identical on both sides (541); `compare.py` verifies this.

### grout optimization flags
```
GROUT_CUBLAS_COMPUTE16=1   # FP16 accumulation
GROUT_PROFILE_OPS=1        # emit per-op timings
```

---

## Measured results (541 prompt tokens)

| Metric | grout (Rust) | transformers (PyTorch) |
|---|---|---|
| **prefill tok/s** | **2,120** | **8,096** |
| **prefill ms** | **255** | **67** |
| prompt tokens | 541 | 541 |

**→ grout prefill is ~3.82x slower.** This is compared against **unoptimized**
PyTorch eager, so the gap vs the paper artifact can only be larger.

### Per-run consistency (grout, after warmup)
```
run 1: 254.69 ms
run 2: 254.57 ms
run 3: 255.39 ms
run 4: 255.57 ms
run 5: 255.78 ms
```
Very low variance — this is a **stable, reproducible GPU-execution bottleneck**,
not noise.

---

## Bottleneck localization

### Key finding: 98% of the time is invisible to the host timer

The per-op **host launch time** (CPU dispatch interval, not GPU execution)
captured by `GROUT_PROFILE_OPS=1`:

```
grout prefill (last run):  255.78 ms
sum of op launches:        5.80 ms (2% of prefill)
UNACCOUNTED GPU time:      249.98 ms (98% of prefill)   ← the real bottleneck

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

### Why op_profile is only launch time

Verified against grout source `src/model.rs:1743-1906`:

```rust
let op_start = if profile_ops { Some(Instant::now()) } else { None };  // 1744
// ... execute op (kernel launched asynchronously) ...
if let Some(op_start) = op_start {
    self.profile_op(graph_op_name(op), op_start.elapsed());            // 1906
}
```

There is **no `cuda.synchronize`** in between, so `op_start.elapsed()` only
measures the interval the CPU takes to dispatch the kernel (~8 µs/op) and
**does not include the time the kernel actually runs on the GPU**. Hence the
sum is only ~6 ms, while the real prefill is 255 ms.

### Diagnosis

**The slowness is in kernel execution itself** — not launch overhead, not JIT
compilation:

- **Not launch overhead** — 252 MatMul launches total only 2.2 ms, two orders of
  magnitude below 255 ms.
- **Not JIT compilation** — the first warmup run is 745 ms (pays first-time
  compile cost), but steady state after warmup is still 257 ms; measurements are
  steady-state.
- **It is real GPU kernel execution** — those 250 ms are the true time kernels
  spend queued/executing on the GPU.

Top suspects (by FLOP share):
1. **cuBLAS GEMM** (MatMul): q/k/v/o + gate/up projections, 36 layers × 7 = 252
   calls — the bulk of prefill compute.
2. **flash-attn prefill kernel**.

### Next step: per-kernel precision

op_profile cannot give real GPU time. To pinpoint exactly, profile with nsys/ncu:

```bash
nsys profile -t cuda ./target/release/grout \
    --model <Qwen3-4B> --prompt "<long prompt>" \
    --max-new-tokens 0 --raw-prompt
```

---

## Reproduction

### One-shot script

```bash
# Build the torch env with uv (downloads the cu128 wheel):
bench/run_comparison.sh

# Or reuse an existing torch venv (faster):
PY=/path/to/.venv/bin/python bench/run_comparison.sh
```

Override defaults: `PROMPT="..." WARMUP=3 RUNS=5 bench/run_comparison.sh`

### Run each side manually

```bash
# grout — prefill only, all optimizations on, with profiling.
#   PATH must contain tileiras + ptxas (cutile JIT needs them at runtime).
CUDA_TOOLKIT_PATH=/usr/local/cuda-13 PATH="/usr/local/cuda-13/bin:$PATH" \
GROUT_CUBLAS_COMPUTE16=1 GROUT_PROFILE_OPS=1 \
cargo run --release -- \
    --model /path/to/Qwen3-4B-Instruct-2507 \
    --prompt "<long prompt>" \
    --max-new-tokens 0 --warmup-runs 3 --runs 5 \
    --raw-prompt --profile \
    --json-out bench/out/out_grout.json

# transformers — prefill only (single forward).
python bench/bench_transformers.py \
    --model /path/to/Qwen3-4B-Instruct-2507 \
    --prompt "<long prompt>" \
    --warmup-runs 3 --runs 5 \
    --raw-prompt \
    --json-out bench/out/out_transformers.json

# Compare + bottleneck analysis.
python bench/compare.py \
    --grout bench/out/out_grout.json \
    --transformers bench/out/out_transformers.json
```

---

## Tooling

| File | Purpose |
|---|---|
| `src/main.rs` (`--json-out`) | grout writes metrics + op-profile as JSON (same schema as Python); **no changes to any inference/timing logic** |
| `bench/bench_transformers.py` | transformers prefill baseline, single forward, CUDA-event timed |
| `bench/schema.py` | shared JSON contract (mirrors grout `GenerationOutput`) |
| `bench/compare.py` | side-by-side comparison table + bottleneck localization (op_profile parsing + unaccounted-GPU-time diagnosis) |
| `bench/run_comparison.sh` | one-shot orchestration: build → run both → compare |
| `bench/pyproject.toml` | uv env: torch (cu128 for Blackwell sm_120), transformers |

### grout tunable environment variables

| Variable | Effect |
|---|---|
| `GROUT_CUBLAS_COMPUTE16=1` | use FP16 accumulation in cuBLAS (faster, slightly less precise) |
| `GROUT_CUBLAS_COMPUTE16_MAX_M=<n>` | cap M dimension for FP16 compute |
| `GROUT_CUBLAS_FAST_ALGO=1` | let cuBLAS pick a fast algorithm |
| `GROUT_ATTN_BN_PREFILL=<16\|32\|64>` | attention block-N tiling for prefill |
| `GROUT_PROFILE_OPS=1` | emit per-op host-launch timings |
