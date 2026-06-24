#!/usr/bin/env python3
"""PyTorch transformers PREFILL baseline that mirrors grout's prefill metric.

For cutile-rs issue #171 we only care about prefill performance (grout prefill
is the slow part), so this script times a SINGLE batched forward over the whole
prompt -- exactly what grout does in src/model.rs:1159 (``step_seq_await``).

Fairness alignment vs grout:
  * dtype       : float16  (grout stores everything as f16; loads bf16 -> f16)
  * attention   : scaled_dot_product_attention (eager, no torch.compile)
  * prefill     : single batched forward over the whole prompt (no chunking)
  * chat template: the literal Qwen3 form grout uses (src/model.rs:1269):
        <|im_start|>user\n{P}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n
  * timing      : CUDA-event timed, torch.cuda.synchronize before reading;
                   warmup runs are NOT counted.

Metric (identical to grout GenerationOutput.prompt_tps, src/model.rs:733):
    prompt_tps = prompt_tokens / prompt_elapsed_s

Usage::

    # uv env (downloads torch cu128 wheel):
    uv run --project bench python bench/bench_transformers.py \
        --model /home/hezhaozhao/models/Qwen3-4B-Instruct-2507 \
        --prompt "Hello, how are you?" \
        --warmup-runs 3 --runs 5 --json-out bench/out/out_transformers.json

    # OR reuse an existing torch venv (faster, no download):
    /path/to/.venv/bin/python bench/bench_transformers.py ...
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Make sibling schema.py importable when run via `python bench/bench_transformers.py`.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from schema import make_run_record, make_summary  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="transformers PREFILL baseline vs grout")
    p.add_argument("--model", required=True, help="path to HF model dir")
    p.add_argument("--prompt", required=True, help="raw input prompt")
    p.add_argument("--max-seq-len", type=int, default=None,
                   help="cap context (defaults to model max_position_embeddings)")
    p.add_argument("--warmup-runs", type=int, default=3,
                   help="untimed forward passes before measuring")
    p.add_argument("--runs", type=int, default=5, help="measured forward passes")
    p.add_argument("--dtype", choices=["half", "float16", "bfloat16"], default="half",
                   help="default float16 to match grout's f16 storage")
    p.add_argument("--raw-prompt", action="store_true",
                   help="skip chat template (use prompt verbatim) like grout --raw-prompt")
    p.add_argument("--json-out", type=Path, default=None,
                   help="write metrics JSON (shared schema with grout)")
    return p.parse_args()


def resolve_dtype(name: str):
    import torch
    return {"half": torch.float16, "float16": torch.float16,
            "bfloat16": torch.bfloat16}[name]


def build_prompt_text(prompt: str, raw: bool) -> str:
    """Build the EXACT prompt text grout uses, so prompt token counts match.

    grout's maybe_apply_chat_template (src/model.rs:1264-1272) formats a raw
    prompt as:
        <|im_start|>user\n{P}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n

    The shipped Qwen3-4B-Instruct-2507 chat template stops at
    `<|im_start|>assistant\n` (no <think> block), so we DON'T call
    apply_chat_template here -- we reproduce grout's literal string. This keeps
    prompt_tokens byte-for-byte identical across both engines.
    """
    if raw or "<|im_start|>" in prompt:
        return prompt
    return (f"<|im_start|>user\n{prompt}<|im_end|>\n"
            f"<|im_start|>assistant\n<think>\n\n</think>\n\n")


def cuda_sync():
    import torch
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def main():
    args = parse_args()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = resolve_dtype(args.dtype)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device_name = torch.cuda.get_device_name(0) if device == "cuda" else "cpu"

    print(f"Loading model from {args.model} (dtype={str(dtype)}) ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    # Load directly from safetensors (low_cpu_mem_usage works without accelerate).
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    ).to(device).eval()

    # Build the exact prompt grout uses (raw flag mirrors grout --raw-prompt).
    prompt_text = build_prompt_text(args.prompt, args.raw_prompt)
    enc = tokenizer(prompt_text, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    attn_mask = enc["attention_mask"].to(device)
    seq_len = int(input_ids.shape[1])

    model_max_len = args.max_seq_len or getattr(
        getattr(model, "config", None), "max_position_embeddings", 10**9)

    print(f"Prompt tokens: {seq_len}  (prefill-only: no decode)")
    print(f"Benchmark mode: warmup_runs={args.warmup_runs}, measured_runs={args.runs}")

    # ---- warmup (untimed) ----
    with torch.inference_mode():
        for w in range(args.warmup_runs):
            t0 = torch.cuda.Event(enable_timing=True) if device == "cuda" else None
            t1 = torch.cuda.Event(enable_timing=True) if device == "cuda" else None
            if t0 is not None:
                t0.record()
            _ = model(input_ids=input_ids, attention_mask=attn_mask, use_cache=False)
            if t1 is not None:
                t1.record()
            cuda_sync()
            ms = t0.elapsed_time(t1) if t0 is not None else 0.0
            tps = seq_len / (ms / 1000.0) if ms > 0 else 0.0
            print(f"warmup {w+1}/{args.warmup_runs}: prefill {ms:.3f} ms "
                  f"({tps:.2f} tok/s, prompt_tokens={seq_len})")

    # ---- measured (prefill only) ----
    run_records = []
    for i in range(args.runs):
        with torch.inference_mode():
            if device == "cuda":
                t0 = torch.cuda.Event(enable_timing=True)
                t1 = torch.cuda.Event(enable_timing=True)
                t0.record()
                _ = model(input_ids=input_ids, attention_mask=attn_mask, use_cache=False)
                t1.record()
                cuda_sync()
                prompt_s = t0.elapsed_time(t1) / 1000.0  # CUDA-event seconds
            else:
                from time import perf_counter
                a = perf_counter()
                _ = model(input_ids=input_ids, attention_mask=attn_mask, use_cache=False)
                cuda_sync()
                prompt_s = perf_counter() - a

        # For prefill-only, decode/total mirror prompt (no decode phase).
        rec = make_run_record(
            i + 1, seq_len, generated_tokens=0,
            prompt_s=prompt_s, decode_s=0.0, total_s=prompt_s,
        )
        run_records.append(rec)
        print(f"run {i+1}/{args.runs}: {rec['prompt_tps']:.2f} prompt tok/s "
              f"(prompt_tokens={seq_len}, prompt_s={rec['prompt_s']:.4f}, "
              f"prompt_ms={rec['prompt_s']*1000:.2f})")

    summary = make_summary(run_records)
    print()
    print(f"summary avg over {args.runs} runs: "
          f"{summary['avg_prompt_tps']:.2f} prompt tok/s "
          f"(avg_prompt_tokens={summary['avg_prompt_tokens']:.1f}, "
          f"avg_prompt_s={summary['avg_prompt_s']:.4f}, "
          f"avg_prompt_ms={summary['avg_prompt_s']*1000:.2f})")

    if args.json_out:
        doc = {
            "engine": "transformers",
            "model": str(args.model),
            "prompt": args.prompt,
            "max_new_tokens": 0,
            "dtype": "float16" if dtype == torch.float16 else "bfloat16",
            "device": device_name,
            "config": {
                "warmup_runs": args.warmup_runs,
                "runs": args.runs,
                "raw_prompt": args.raw_prompt,
                "attn": "sdpa",
                "prefill_only": True,
                "torch_version": torch.__version__,
            },
            "runs": run_records,
            "summary": summary,
            "op_profile": [],  # transformers side has no per-op breakdown
        }
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(doc, indent=2))
        print(f"\nWrote JSON metrics to {args.json_out}")


if __name__ == "__main__":
    main()
