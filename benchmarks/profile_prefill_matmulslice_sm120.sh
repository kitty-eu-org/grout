#!/usr/bin/env bash
# Profile Grout prefill at long prompt lengths with the RTX 5090 / sm_120
# long-prefill parameter profile, then summarize per-op time shares.
#
# Default use:
#   ./benchmarks/profile_prefill_matmulslice_sm120.sh
#
# Useful overrides:
#   MODEL_HF=/path/to/Qwen3-4B PP_VALUES="8192 32768" REPS=1 WARMUP_REPS=0 \
#     ./benchmarks/profile_prefill_matmulslice_sm120.sh
#
# The op profile uses stream synchronization after each op. That is intentional:
# it gives an issue-friendly breakdown showing whether MatMulSlice or Attention
# dominates at 8192 vs 32768.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GROUT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

default_model="$GROUT_DIR/../hf_models/qwen3_4b"
if [[ ! -d "$default_model" && -d /home/hezhaozhao/kaio-vllm/models/Qwen3-4B ]]; then
    default_model=/home/hezhaozhao/kaio-vllm/models/Qwen3-4B
fi
MODEL_HF="${MODEL_HF:-$default_model}"

CUDA_BIN="${CUDA_BIN:-/usr/local/cuda-13.3/bin}"
if [[ -d "$CUDA_BIN" ]]; then
    export PATH="$CUDA_BIN:$PATH"
    export CUDA_HOME="${CUDA_HOME:-${CUDA_BIN%/bin}}"
fi

read -r -a PP_ARRAY <<< "${PP_VALUES:-8192 32768}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1}"
REPS="${REPS:-1}"
WARMUP_REPS="${WARMUP_REPS:-0}"
BUILD="${BUILD:-1}"
TS="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${OUT_DIR:-$SCRIPT_DIR/results/prefill_matmulslice_sm120/$TS}"
PROMPTS_DIR="$OUT_DIR/prompts"
SUMMARY_MD="$OUT_DIR/summary.md"
mkdir -p "$PROMPTS_DIR"

log() {
    printf '\n\033[1;36m==> %s\033[0m\n' "$*"
}

find_python_with_transformers() {
    local candidates=(
        "${PYTHON:-}"
        "$GROUT_DIR/../bench_envs/vllm_env/bin/python3"
        "$GROUT_DIR/bench/.venv/bin/python"
        "$GROUT_DIR/.venv/bin/python"
        "/home/hezhaozhao/code/vllms/.venv/bin/python"
        "/home/hezhaozhao/openinfer/.venv/bin/python"
        "python3"
    )
    local py
    for py in "${candidates[@]}"; do
        [[ -n "$py" ]] || continue
        if command -v "$py" >/dev/null 2>&1 \
            && "$py" -c 'import transformers' >/dev/null 2>&1; then
            command -v "$py"
            return 0
        fi
    done
    return 1
}

PY_WITH_TRANSFORMERS="$(find_python_with_transformers || true)"
if [[ -z "$PY_WITH_TRANSFORMERS" ]]; then
    echo "ERROR: could not find a Python with transformers installed." >&2
    echo "Set PYTHON=/path/to/python, or create one with uv and install transformers." >&2
    exit 2
fi

log "Configuration"
cat <<EOF
model          : $MODEL_HF
pp values      : ${PP_ARRAY[*]}
max_new_tokens : $MAX_NEW_TOKENS
reps/warmup    : $REPS / $WARMUP_REPS
python         : $PY_WITH_TRANSFORMERS
cuda bin       : ${CUDA_BIN:-none}
out dir        : $OUT_DIR
EOF

if [[ "$BUILD" == "1" ]]; then
    log "Building grout_bench"
    (cd "$GROUT_DIR" && cargo build --release --features benchmarks --bin grout_bench)
fi

log "Generating exact-token prompts"
"$PY_WITH_TRANSFORMERS" "$SCRIPT_DIR/make_prompts.py" \
    --model "$MODEL_HF" \
    --out-dir "$PROMPTS_DIR" \
    --pp "${PP_ARRAY[@]}" \
    --chat-template-pp 0

# 5090 / sm_120 profile from benchmarks/sweep_pp_sm120.sh. For pp=32768 this
# deliberately applies the pp=8192 long-prefill values as plain env vars,
# because sweep_pp_sm120.sh only defines PP_2048 and PP_8192 overrides.
: "${GROUT_TUNING_PROFILE:=sm120_5090}"
: "${GROUT_CUBLAS_COMPUTE16:=1}"
: "${GROUT_CUDA_GRAPH_DECODE:=1}"
: "${GROUT_FLASH_DECODE:=0}"
: "${GROUT_FMHA_SPLIT_KV:=1}"
: "${GROUT_FMHA_DECODE_LATENCY:=4}"
: "${GROUT_FMHA_DECODE_OCCUPANCY:=2}"
: "${GROUT_FMHA_MERGE_CHUNK_D:=16}"
: "${GROUT_FMHA_MERGE_LATENCY:=2}"
: "${GROUT_FUSED_QK_ROPE_KV_DECODE:=1}"
: "${GROUT_QK_ROPE_LATENCY:=2}"
: "${GROUT_QK_ROPE_OCCUPANCY:=1}"
: "${GROUT_QK_ROPE_CGA:=0}"
: "${GROUT_KV_CACHE_DYN_CHUNK_D:=32}"
: "${GROUT_EMBED_BLOCK:=1024}"
: "${GROUT_RMS_BLOCK:=4096}"
: "${GROUT_ARGMAX_BLOCK:=128}"
: "${GROUT_KV_CACHE_BM_S:=16}"
: "${GROUT_FUSED_QK_ROPE_KV_PREFILL:=1}"
: "${GROUT_ATTN_BM_PREFILL:=16}"
: "${GROUT_ATTN_BN_PREFILL:=64}"
: "${GROUT_FMHA_PREFILL:=1}"
: "${GROUT_FMHA_PREFILL_GQA:=0}"
: "${GROUT_FMHA_PREFILL_GQA_LPT:=1}"
: "${GROUT_FMHA_PREFILL_GQA_GROUP:=0}"
: "${GROUT_FMHA_PREFILL_LPT_SWIZZLE:=8}"
: "${GROUT_FMHA_PREFILL_LPT_SCHED:=1}"
: "${GROUT_FMHA_PREFILL_LPT_MASK_SPLIT:=0}"
: "${GROUT_FMHA_PREFILL_LATENCY:=4}"
: "${GROUT_FMHA_PREFILL_OCCUPANCY:=2}"
: "${GROUT_ATTN_BN_DECODE:=64}"
: "${GROUT_FMHA_NUM_KV_SPLITS:=32}"

export \
    MODEL_HF \
    MAX_NEW_TOKENS \
    REPS \
    WARMUP_REPS \
    GROUT_TUNING_PROFILE \
    GROUT_CUBLAS_COMPUTE16 \
    GROUT_CUDA_GRAPH_DECODE \
    GROUT_FLASH_DECODE \
    GROUT_FMHA_SPLIT_KV \
    GROUT_FMHA_DECODE_LATENCY \
    GROUT_FMHA_DECODE_OCCUPANCY \
    GROUT_FMHA_MERGE_CHUNK_D \
    GROUT_FMHA_MERGE_LATENCY \
    GROUT_FUSED_QK_ROPE_KV_DECODE \
    GROUT_QK_ROPE_LATENCY \
    GROUT_QK_ROPE_OCCUPANCY \
    GROUT_QK_ROPE_CGA \
    GROUT_KV_CACHE_DYN_CHUNK_D \
    GROUT_EMBED_BLOCK \
    GROUT_RMS_BLOCK \
    GROUT_ARGMAX_BLOCK \
    GROUT_KV_CACHE_BM_S \
    GROUT_FUSED_QK_ROPE_KV_PREFILL \
    GROUT_ATTN_BM_PREFILL \
    GROUT_ATTN_BN_PREFILL \
    GROUT_FMHA_PREFILL \
    GROUT_FMHA_PREFILL_GQA \
    GROUT_FMHA_PREFILL_GQA_LPT \
    GROUT_FMHA_PREFILL_GQA_GROUP \
    GROUT_FMHA_PREFILL_LPT_SWIZZLE \
    GROUT_FMHA_PREFILL_LPT_SCHED \
    GROUT_FMHA_PREFILL_LPT_MASK_SPLIT \
    GROUT_FMHA_PREFILL_LATENCY \
    GROUT_FMHA_PREFILL_OCCUPANCY \
    GROUT_ATTN_BN_DECODE \
    GROUT_FMHA_NUM_KV_SPLITS

run_one_pp() {
    local pp="$1"
    local max_seq_len_var="MAX_SEQ_LEN_PP_${pp}"
    local max_seq_len="${!max_seq_len_var:-$((pp + MAX_NEW_TOKENS))}"
    local log_path="$OUT_DIR/grout_pp_${pp}.log"

    log "Profiling pp=${pp}"
    (
        cd "$GROUT_DIR"
        env \
            GROUT_PROFILE=1 \
            GROUT_PROFILE_OPS=1 \
            GROUT_PROFILE_SYNC_OPS=1 \
            GROUT_TUNING_PROFILE="$GROUT_TUNING_PROFILE" \
            GROUT_CUBLAS_COMPUTE16="$GROUT_CUBLAS_COMPUTE16" \
            GROUT_CUDA_GRAPH_DECODE="$GROUT_CUDA_GRAPH_DECODE" \
            GROUT_FLASH_DECODE="$GROUT_FLASH_DECODE" \
            GROUT_FMHA_SPLIT_KV="$GROUT_FMHA_SPLIT_KV" \
            GROUT_FMHA_DECODE_LATENCY="$GROUT_FMHA_DECODE_LATENCY" \
            GROUT_FMHA_DECODE_OCCUPANCY="$GROUT_FMHA_DECODE_OCCUPANCY" \
            GROUT_FMHA_MERGE_CHUNK_D="$GROUT_FMHA_MERGE_CHUNK_D" \
            GROUT_FMHA_MERGE_LATENCY="$GROUT_FMHA_MERGE_LATENCY" \
            GROUT_FUSED_QK_ROPE_KV_DECODE="$GROUT_FUSED_QK_ROPE_KV_DECODE" \
            GROUT_QK_ROPE_LATENCY="$GROUT_QK_ROPE_LATENCY" \
            GROUT_QK_ROPE_OCCUPANCY="$GROUT_QK_ROPE_OCCUPANCY" \
            GROUT_QK_ROPE_CGA="$GROUT_QK_ROPE_CGA" \
            GROUT_KV_CACHE_DYN_CHUNK_D="$GROUT_KV_CACHE_DYN_CHUNK_D" \
            GROUT_EMBED_BLOCK="$GROUT_EMBED_BLOCK" \
            GROUT_RMS_BLOCK="$GROUT_RMS_BLOCK" \
            GROUT_ARGMAX_BLOCK="$GROUT_ARGMAX_BLOCK" \
            GROUT_KV_CACHE_BM_S="$GROUT_KV_CACHE_BM_S" \
            GROUT_FUSED_QK_ROPE_KV_PREFILL="$GROUT_FUSED_QK_ROPE_KV_PREFILL" \
            GROUT_ATTN_BM_PREFILL="$GROUT_ATTN_BM_PREFILL" \
            GROUT_ATTN_BN_PREFILL="$GROUT_ATTN_BN_PREFILL" \
            GROUT_FMHA_PREFILL="$GROUT_FMHA_PREFILL" \
            GROUT_FMHA_PREFILL_GQA="$GROUT_FMHA_PREFILL_GQA" \
            GROUT_FMHA_PREFILL_GQA_LPT="$GROUT_FMHA_PREFILL_GQA_LPT" \
            GROUT_FMHA_PREFILL_GQA_GROUP="$GROUT_FMHA_PREFILL_GQA_GROUP" \
            GROUT_FMHA_PREFILL_LPT_SWIZZLE="$GROUT_FMHA_PREFILL_LPT_SWIZZLE" \
            GROUT_FMHA_PREFILL_LPT_SCHED="$GROUT_FMHA_PREFILL_LPT_SCHED" \
            GROUT_FMHA_PREFILL_LPT_MASK_SPLIT="$GROUT_FMHA_PREFILL_LPT_MASK_SPLIT" \
            GROUT_FMHA_PREFILL_LATENCY="$GROUT_FMHA_PREFILL_LATENCY" \
            GROUT_FMHA_PREFILL_OCCUPANCY="$GROUT_FMHA_PREFILL_OCCUPANCY" \
            GROUT_ATTN_BN_DECODE="$GROUT_ATTN_BN_DECODE" \
            GROUT_FMHA_NUM_KV_SPLITS="$GROUT_FMHA_NUM_KV_SPLITS" \
            ./target/release/grout_bench \
                --model "$MODEL_HF" \
                --prompt-file "$PROMPTS_DIR/pp_${pp}.txt" \
                --raw-prompt \
                --max-new-tokens "$MAX_NEW_TOKENS" \
                --max-seq-len "$max_seq_len" \
                --reps "$REPS" \
                --warmup-reps "$WARMUP_REPS" \
                --profile \
                --ignore-eos
    ) 2>&1 | tee "$log_path"
}

for pp in "${PP_ARRAY[@]}"; do
    run_one_pp "$pp"
done

log "Writing Markdown summary"
"$PY_WITH_TRANSFORMERS" - "$OUT_DIR" "$SUMMARY_MD" "${PP_ARRAY[@]}" <<'PY'
from __future__ import annotations

import datetime as dt
import os
import re
import sys
from pathlib import Path

out_dir = Path(sys.argv[1])
summary_path = Path(sys.argv[2])
pps = [int(x) for x in sys.argv[3:]]

OP_RE = re.compile(
    r"^\s{2}(\S+)\s+total_ms=\s*([0-9.]+)\s+avg_us=\s*([0-9.]+)\s+calls=(\d+)"
)


def parse_log(pp: int) -> dict:
    path = out_dir / f"grout_pp_{pp}.log"
    text = path.read_text(errors="replace")
    timed = [float(x) for x in re.findall(r"\[timed\].*?prefill_ms=([0-9.]+)", text)]
    prompt_tokens = None
    m = re.search(r"\[timed\].*?prompt_tokens=(\d+)", text)
    if m:
        prompt_tokens = int(m.group(1))
    profile_avg = None
    m = re.search(r"avg_ms: prefill=([0-9.]+)", text)
    if m:
        profile_avg = float(m.group(1))
    ops = {}
    for line in text.splitlines():
        m = OP_RE.match(line)
        if not m:
            continue
        name, total_ms, avg_us, calls = m.groups()
        ops[name] = {
            "total_ms": float(total_ms),
            "avg_us": float(avg_us),
            "calls": int(calls),
        }
    prefill_ms = profile_avg or (sum(timed) / len(timed) if timed else 0.0)
    return {
        "pp": pp,
        "path": path,
        "prompt_tokens": prompt_tokens,
        "timed_prefill_ms": timed,
        "prefill_ms": prefill_ms,
        "ops": ops,
    }


def pct(ms: float, total: float) -> float:
    return 100.0 * ms / total if total > 0 else 0.0


data = [parse_log(pp) for pp in pps]
by_pp = {d["pp"]: d for d in data}
all_ops = sorted(
    {name for d in data for name in d["ops"]},
    key=lambda name: max((d["ops"].get(name, {}).get("total_ms", 0.0) for d in data)),
    reverse=True,
)

lines = []
lines.append("# Grout 5090/sm120 Prefill MatMulSlice Profile")
lines.append("")
lines.append(f"- Generated: {dt.datetime.now().isoformat(timespec='seconds')}")
lines.append(f"- Model: `{os.environ.get('MODEL_HF', '')}`")
lines.append(f"- Prompt lengths: `{', '.join(map(str, pps))}`")
lines.append(f"- `max_new_tokens`: `{os.environ.get('MAX_NEW_TOKENS', '1')}`")
lines.append(f"- `reps/warmup`: `{os.environ.get('REPS', '1')}/{os.environ.get('WARMUP_REPS', '0')}`")
lines.append("- Note: op profiling uses `GROUT_PROFILE_SYNC_OPS=1`, so each op is stream-synchronized for attribution.")
lines.append("")

lines.append("## Timing Summary")
lines.append("")
lines.append("| pp | prompt_tokens | prefill_ms | prefill_s | MatMulSlice ms | MatMulSlice % | Attention ms | Attention % |")
lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|")
for d in data:
    total = d["prefill_ms"]
    mm = d["ops"].get("MatMulSlice", {}).get("total_ms", 0.0)
    attn = d["ops"].get("Attention", {}).get("total_ms", 0.0)
    lines.append(
        f"| {d['pp']} | {d['prompt_tokens'] or ''} | {total:.3f} | {total / 1000.0:.3f} "
        f"| {mm:.3f} | {pct(mm, total):.1f}% | {attn:.3f} | {pct(attn, total):.1f}% |"
    )

if len(data) >= 2:
    base = data[0]
    lines.append("")
    lines.append("## Slowdown Versus First PP")
    lines.append("")
    lines.append("| pp | prefill slowdown | MatMulSlice slowdown | Attention slowdown |")
    lines.append("|---:|---:|---:|---:|")
    base_total = base["prefill_ms"]
    base_mm = base["ops"].get("MatMulSlice", {}).get("total_ms", 0.0)
    base_attn = base["ops"].get("Attention", {}).get("total_ms", 0.0)
    for d in data:
        total = d["prefill_ms"]
        mm = d["ops"].get("MatMulSlice", {}).get("total_ms", 0.0)
        attn = d["ops"].get("Attention", {}).get("total_ms", 0.0)
        lines.append(
            f"| {d['pp']} | {total / base_total if base_total else 0.0:.2f}x "
            f"| {mm / base_mm if base_mm else 0.0:.2f}x "
            f"| {attn / base_attn if base_attn else 0.0:.2f}x |"
        )

lines.append("")
lines.append("## Per-Op Breakdown")
lines.append("")
header = "| op |" + "".join(f" pp={d['pp']} ms | pp={d['pp']} % | calls |" for d in data)
sep = "|---|" + "".join("---:|---:|---:|" for _ in data)
lines.append(header)
lines.append(sep)
for op in all_ops:
    row = f"| `{op}` |"
    for d in data:
        entry = d["ops"].get(op)
        if entry:
            row += (
                f" {entry['total_ms']:.3f} |"
                f" {pct(entry['total_ms'], d['prefill_ms']):.1f}% |"
                f" {entry['calls']} |"
            )
        else:
            row += "  |  |  |"
    lines.append(row)

lines.append("")
lines.append("## Parameters")
lines.append("")
param_names = [
    "GROUT_CUBLAS_COMPUTE16",
    "GROUT_KV_CACHE_BM_S",
    "GROUT_FUSED_QK_ROPE_KV_PREFILL",
    "GROUT_ATTN_BM_PREFILL",
    "GROUT_ATTN_BN_PREFILL",
    "GROUT_FMHA_PREFILL",
    "GROUT_FMHA_PREFILL_GQA",
    "GROUT_FMHA_PREFILL_GQA_LPT",
    "GROUT_FMHA_PREFILL_GQA_GROUP",
    "GROUT_FMHA_PREFILL_LPT_SWIZZLE",
    "GROUT_FMHA_PREFILL_LPT_SCHED",
    "GROUT_FMHA_PREFILL_LPT_MASK_SPLIT",
    "GROUT_FMHA_PREFILL_LATENCY",
    "GROUT_FMHA_PREFILL_OCCUPANCY",
]
lines.append("| env | value |")
lines.append("|---|---:|")
for name in param_names:
    lines.append(f"| `{name}` | `{os.environ.get(name, '')}` |")

lines.append("")
lines.append("## Raw Logs")
lines.append("")
for d in data:
    lines.append(f"- pp={d['pp']}: `{d['path'].name}`")

summary = "\n".join(lines) + "\n"
summary_path.write_text(summary)
print(summary)
PY

log "Done"
echo "Summary: $SUMMARY_MD"
