#!/usr/bin/env bash
#
# PREFILL comparison: grout (Rust) vs transformers (PyTorch) on Qwen3-4B.
#
# Focuses on prefill only (the slow part per cutile-rs issue #171). Runs both
# engines on the SAME prompt with matching metrics (see schema.py), then prints
# a side-by-side prefill comparison plus grout's prefill bottleneck analysis.
#
# Usage:
#   bench/run_comparison.sh
#
# Override defaults via environment variables:
#   MODEL=... PROMPT="..." WARMUP=3 RUNS=5 ./bench/run_comparison.sh
#
# Python interpreter: by default uses `uv run --project bench`. To reuse an
# existing torch venv (faster, no download), set PY=python or PY=/path/.venv/bin/python.
set -euo pipefail

# Resolve repo root (this script lives in bench/).
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MODEL="${MODEL:-/home/hezhaozhao/models/Qwen3-4B-Instruct-2507}"
PROMPT="${PROMPT:-Hello, how are you? Please introduce yourself in two sentences.}"
WARMUP="${WARMUP:-3}"
RUNS="${RUNS:-5}"

OUT_DIR="${ROOT}/bench/out"
mkdir -p "$OUT_DIR"
GROUT_JSON="$OUT_DIR/out_grout.json"
TF_JSON="$OUT_DIR/out_transformers.json"

# Python runner: uv env by default, or an explicit interpreter via PY=...
if [[ -n "${PY:-}" ]]; then
  PY_RUN=("$PY")
else
  PY_RUN=(uv run --project "$ROOT/bench" python)
fi

echo "============================================================"
echo " PREFILL-ONLY comparison"
echo " Model:  $MODEL"
echo " Prompt: $PROMPT"
echo " runs:   warmup=$WARMUP measured=$RUNS"
echo "============================================================"

# ---------------------------------------------------------------------------
# 1) grout (Rust) — prefill only (--max-new-tokens 0), optimizations ON,
#    per-op profiling enabled so we can find the bottleneck.
# ---------------------------------------------------------------------------
echo
echo "[1/3] Running grout (Rust) — prefill only ..."
# PATH must contain tileiras + ptxas (cutile JIT needs them at runtime).
CUDA_TOOLKIT_PATH="${CUDA_TOOLKIT_PATH:-/usr/local/cuda-13}" \
PATH="/usr/local/cuda-13/bin:${PATH}" \
GROUT_CUBLAS_COMPUTE16=1 \
GROUT_PROFILE_OPS=1 \
cargo run --release -- \
    --model "$MODEL" \
    --prompt "$PROMPT" \
    --max-new-tokens 0 \
    --warmup-runs "$WARMUP" --runs "$RUNS" \
    --raw-prompt \
    --profile \
    --json-out "$GROUT_JSON"

# ---------------------------------------------------------------------------
# 2) transformers (PyTorch) — prefill only (single forward, eager + SDPA, fp16).
# ---------------------------------------------------------------------------
echo
echo "[2/3] Running transformers (PyTorch) — prefill only ..."
"${PY_RUN[@]}" "$ROOT/bench/bench_transformers.py" \
    --model "$MODEL" \
    --prompt "$PROMPT" \
    --warmup-runs "$WARMUP" --runs "$RUNS" \
    --raw-prompt \
    --json-out "$TF_JSON"

# ---------------------------------------------------------------------------
# 3) Compare + bottleneck analysis.
# ---------------------------------------------------------------------------
echo
echo "[3/3] Comparing ..."
"${PY_RUN[@]}" "$ROOT/bench/compare.py" \
    --grout "$GROUT_JSON" --transformers "$TF_JSON"

echo
echo "Done. JSON outputs in $OUT_DIR"
