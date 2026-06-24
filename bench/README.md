# grout vs transformers — PREFILL benchmark

Compares the prefill (prompt-processing) performance of the Rust **grout** engine
(built on [cutile-rs](https://github.com/NVlabs/cutile-rs)) against a **PyTorch
transformers** baseline, for [cutile-rs issue #171][issue] (grout prefill is
much slower than the paper artifact on RTX 5070 Ti).

> **Prefill only.** Decode is out of scope — issue #171 is about prefill.
> grout runs with `--max-new-tokens 0`; transformers runs a single forward.

Both sides produce the **same JSON metric schema** (`bench/schema.py`), run the
same warmup + measurement loop, and use the same dtype / prompt / chat template,
so the comparison is apples-to-apples. The grout side additionally emits a
per-op profile so you can see **where** prefill time goes.

[issue]: https://github.com/NVlabs/cutile-rs/issues/171#issuecomment-4781966468

---

## Quick start

```bash
# One-shot: builds grout, sets up the torch env, runs both, prints comparison.
bench/run_comparison.sh
```

Reuse an existing torch venv (much faster — no wheel download):

```bash
PY=/home/hezhaozhao/OCRFlux_speculative/.venv/bin/python bench/run_comparison.sh
```

Override defaults with env vars:

```bash
PROMPT="Explain gradient descent in one paragraph." WARMUP=3 RUNS=5 \
  bench/run_comparison.sh
```

Outputs land in `bench/out/` (`out_grout.json`, `out_transformers.json`).

---

## What it measures

| metric | definition |
|--------|-----------|
| `prompt_tps` | `prompt_tokens / prompt_elapsed_s` |
| `prompt_s` / `prompt_ms` | absolute prefill wall-time |

(identical to grout's `GenerationOutput.prompt_tps`, `src/model.rs:733`.)

- **Warmup runs are not counted** (grout `--warmup-runs`; transformers `--warmup-runs`).
- Timers wrap the GPU forward and synchronize before reading.

## Fairness alignment

| dimension | grout (Rust) | transformers (PyTorch) |
|-----------|--------------|------------------------|
| dtype | f16 (loads bf16 → f16) | `torch.float16` |
| attention | flash-attn kernel | `scaled_dot_product_attention` (eager, no `torch.compile`) |
| prefill | single batched forward (`step_seq_await`) | single batched forward |
| chat template | Qwen3 `enable_thinking=False` literal | the *same* literal string (see below) |

The Qwen3-4B-Instruct-2507 tokenizer's `apply_chat_template` stops at
`<|im_start|>assistant\n` without a `<think>` block, whereas grout always appends
`<think>\n\n</think>\n\n`. So the transformers side reproduces grout's **literal**
prompt string to keep token counts byte-identical. `compare.py` verifies this.

## Finding the bottleneck

`compare.py` section 3 parses grout's `op_profile` (produced by
`GROUT_PROFILE_OPS=1` + `--profile`, both set by `run_comparison.sh`) and ranks
ops by prefill time. The dominant op is what you attack:

| dominant op | likely cause | knob to try |
|-------------|--------------|-------------|
| `MatMul` | prefill GEMM (q/k/v/o + gate/up) | `GROUT_CUBLAS_COMPUTE16=1`, `GROUT_CUBLAS_FAST_ALGO=1` |
| `Attention` | flash-attn kernel / tiling | `GROUT_ATTN_BN_PREFILL=16\|32\|64` |
| `RmsNorm` / `Rope` | launch overhead | reduce per-layer kernel count |
| `KvCacheUpdate` | KV write bandwidth | check memory layout |

> The op-profile timings are **host-side launch durations** (CPU wall clock
> between dispatches), not CUDA events. Treat absolute ms as approximate, but
> the **ranking and ratios** reliably identify the bottleneck.

---

## Files

| file | purpose |
|------|---------|
| `bench/pyproject.toml` | uv env: torch (cu128 for Blackwell sm_120), transformers |
| `bench/bench_transformers.py` | PyTorch prefill baseline, writes the shared JSON schema |
| `bench/schema.py` | shared metric definitions (mirrors `src/model.rs:722`) |
| `bench/compare.py` | prefill comparison table + grout bottleneck analysis |
| `bench/run_comparison.sh` | orchestrates build → run both → compare |
| `src/main.rs` (`--json-out`) | grout writes the same JSON schema; parses its op-profile |

## Running each side manually

```bash
# grout — prefill only, all optimizations on, with profiling.
#   PATH must contain tileiras + ptxas (cutile JIT needs them at runtime).
CUDA_TOOLKIT_PATH=/usr/local/cuda-13 PATH="/usr/local/cuda-13/bin:$PATH" \
GROUT_CUBLAS_COMPUTE16=1 GROUT_PROFILE_OPS=1 \
cargo run --release -- \
    --model /home/hezhaozhao/models/Qwen3-4B-Instruct-2507 \
    --prompt "Hello, how are you?" \
    --max-new-tokens 0 --warmup-runs 3 --runs 5 \
    --raw-prompt --profile \
    --json-out bench/out/out_grout.json

# transformers — prefill only (single forward).
uv run --project bench python bench/bench_transformers.py \
    --model /home/hezhaozhao/models/Qwen3-4B-Instruct-2507 \
    --prompt "Hello, how are you?" \
    --warmup-runs 3 --runs 5 \
    --raw-prompt \
    --json-out bench/out/out_transformers.json

# compare
uv run --project bench python bench/compare.py \
    --grout bench/out/out_grout.json \
    --transformers bench/out/out_transformers.json
```
