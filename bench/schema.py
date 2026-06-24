"""Shared JSON contract between grout (Rust) and the PyTorch transformers baseline.

Both ``src/main.rs --json-out`` and ``bench/bench_transformers.py --json-out``
write a JSON document matching this shape so that ``bench/compare.py`` can
diff them apples-to-apples.

Top-level keys
--------------
engine          : "grout" | "transformers"
model           : path to the model directory
prompt          : the (raw) prompt string fed in (before chat template)
max_new_tokens  : int
dtype           : "float16"   (both sides run fp16 to match)
device          : GPU name, e.g. "NVIDIA GeForce RTX 5070 Ti"
config          : dict with run knobs (warmup_runs, runs, engine-specific flags)
runs            : list of per-run records (one per measured run)
summary         : averaged metrics over ``runs``
op_profile      : list of {op,total_ms,avg_us,calls}  (grout only, when profiled)

A per-run record looks like::

    {
      "run": 1,
      "prompt_tokens": 42,
      "generated_tokens": 128,
      "prompt_s": 0.0123,        # absolute prefill wall-time (seconds)
      "decode_s": 1.456,
      "total_s": 1.468,
      "prompt_tps": 3414.6,      # prompt_tokens / prompt_s
      "decode_tps": 87.9,        # generated_tokens / decode_s
      "total_tps": 87.2          # generated_tokens / total_s
    }

These mirror grout's ``GenerationOutput`` (src/model.rs:722) exactly:
    prompt_tps = prompt_tokens / prompt_elapsed
    decode_tps = generated_tokens / decode_elapsed
    total_tps  = generated_tokens / total_elapsed
"""

from __future__ import annotations

from typing import Any

PER_RUN_KEYS = (
    "run",
    "prompt_tokens",
    "generated_tokens",
    "prompt_s",
    "decode_s",
    "total_s",
    "prompt_tps",
    "decode_tps",
    "total_tps",
)

SUMMARY_KEYS = (
    "avg_prompt_tps",
    "avg_decode_tps",
    "avg_total_tps",
    "avg_prompt_s",
    "avg_decode_s",
    "avg_total_s",
    "avg_prompt_tokens",
    "avg_generated_tokens",
)


def make_run_record(
    index: int,
    prompt_tokens: int,
    generated_tokens: int,
    prompt_s: float,
    decode_s: float,
    total_s: float,
) -> dict[str, Any]:
    """Build a single per-run record with the grout-compatible metric defs."""
    prompt_tps = prompt_tokens / max(prompt_s, 1e-9)
    decode_tps = generated_tokens / max(decode_s, 1e-9)
    total_tps = generated_tokens / max(total_s, 1e-9)
    return {
        "run": index,
        "prompt_tokens": int(prompt_tokens),
        "generated_tokens": int(generated_tokens),
        "prompt_s": prompt_s,
        "decode_s": decode_s,
        "total_s": total_s,
        "prompt_tps": prompt_tps,
        "decode_tps": decode_tps,
        "total_tps": total_tps,
    }


def make_summary(run_records: list[dict[str, Any]]) -> dict[str, float]:
    """Average a list of per-run records into a summary dict (mirrors grout)."""
    n = len(run_records)
    if n == 0:
        return {k: 0.0 for k in SUMMARY_KEYS}
    keys = ("prompt_tps", "decode_tps", "total_tps", "prompt_s", "decode_s", "total_s")
    sums = {k: sum(r[k] for r in run_records) for k in keys}
    return {
        "avg_prompt_tps": sums["prompt_tps"] / n,
        "avg_decode_tps": sums["decode_tps"] / n,
        "avg_total_tps": sums["total_tps"] / n,
        "avg_prompt_s": sums["prompt_s"] / n,
        "avg_decode_s": sums["decode_s"] / n,
        "avg_total_s": sums["total_s"] / n,
        "avg_prompt_tokens": sum(r["prompt_tokens"] for r in run_records) / n,
        "avg_generated_tokens": sum(r["generated_tokens"] for r in run_records) / n,
    }
