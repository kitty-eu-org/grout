#!/usr/bin/env python3
"""Compare grout (Rust) vs transformers (PyTorch) PREFILL performance and
locate grout's prefill bottleneck.

For cutile-rs issue #171 we focus on prefill (the slow part). Reads the two
``--json-out`` documents produced by:

  * ``src/main.rs``                  -> engine = "grout"
  * ``bench/bench_transformers.py``  -> engine = "transformers"  (prefill-only)

Both follow the shared schema in ``bench/schema.py``.

Output sections
---------------
1. Config sanity check (dtype / prompt tokens must match).
2. Prefill comparison: prompt tok/s and prompt ms, with a slowdown ratio.
3. Bottleneck analysis: parses grout's ``op_profile`` (from ``--profile`` +
   ``GROUT_PROFILE_OPS=1``) to rank prefill ops by time and show their share.
4. Tuning hints (env vars to try).

Usage::

    uv run --project bench python bench/compare.py \
        --grout bench/out/out_grout.json --transformers bench/out/out_transformers.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="grout vs transformers PREFILL comparison")
    p.add_argument("--grout", type=Path, required=True, help="grout metrics JSON")
    p.add_argument("--transformers", type=Path, required=True,
                   help="transformers metrics JSON")
    p.add_argument("--top", type=int, default=8,
                   help="top-N ops to show in bottleneck analysis")
    return p.parse_args()


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def hr(char: str = "=", width: int = 78) -> str:
    return char * width


def fmt(v, w=12, prec=2) -> str:
    if v is None:
        return f"{'n/a':>{w}}"
    if isinstance(v, float):
        return f"{v:>{w},.{prec}f}"
    return f"{str(v):>{w}}"


def section(title: str, char: str = "="):
    print()
    print(hr(char))
    print(title)
    print(hr(char))


def check_config(g: dict, t: dict):
    section("1. Config sanity check", "-")

    def short(s, n=40):
        if not isinstance(s, str):
            return s
        return s if len(s) <= n else s[:n] + "..."

    pairs = [
        ("dtype", g.get("dtype"), t.get("dtype")),
        ("device", g.get("device"), t.get("device")),
        ("prompt (raw)", short(g.get("prompt")), short(t.get("prompt"))),
    ]
    aligned = True
    for name, gv, tv in pairs:
        same = (gv == tv)
        if not same:
            aligned = False
        flag = "ok " if same else "DIFF"
        print(f"  [{flag}] {name:<16} grout={gv!r:<26} transformers={tv!r}")
    # prompt_tokens alignment (post-template token counts) -- the key fairness check.
    gp = g.get("summary", {}).get("avg_prompt_tokens")
    tp = t.get("summary", {}).get("avg_prompt_tokens")
    same_pt = gp is not None and tp is not None and abs(gp - tp) <= 1
    flag = "ok " if same_pt else "DIFF"
    print(f"  [{flag}] {'prompt_tokens':<16} grout={gp}  transformers={tp}")
    if not aligned or not same_pt:
        print("\n  WARNING: configs or prompt token counts differ -> comparison "
              "is NOT fair. Make sure both sides apply the identical Qwen3 chat "
              "template (grout's maybe_apply_chat_template).")
    else:
        print("\n  prompt token counts match -> fair prefill comparison.")


def prefill_comparison(g: dict, t: dict):
    section("2. Prefill comparison (grout vs transformers)", "-")
    gs = g.get("summary", {})
    ts = t.get("summary", {})
    gc = g.get("config", {})
    tc = t.get("config", {})
    print(f"  grout:        runs={gc.get('runs')} cuda_graph={gc.get('cuda_graph_decode')} "
          f"cublas_fp16={gc.get('cublas_compute16')}")
    print(f"  transformers: runs={tc.get('runs')} attn={tc.get('attn','sdpa')} "
          f"torch={tc.get('torch_version','?')}")

    # Both tps (tok/s) and ms (absolute), since issue #171 reports absolute time.
    g_tps = gs.get("avg_prompt_tps")
    t_tps = ts.get("avg_prompt_tps")
    g_ms = (gs.get("avg_prompt_s") or 0.0) * 1000.0
    t_ms = (ts.get("avg_prompt_s") or 0.0) * 1000.0

    print()
    print(f"  {'metric':<22}{'grout':>14}{'transformers':>15}")
    print(f"  {'-'*22}{'-'*14}{'-'*15}")
    print(f"  {'prompt tok/s':<22}{fmt(g_tps,14)}{fmt(t_tps,15)}")
    print(f"  {'prompt ms':<22}{fmt(g_ms,14,3)}{fmt(t_ms,15,3)}")
    print(f"  {'prompt tokens':<22}"
          f"{fmt(gs.get('avg_prompt_tokens'),14,0)}{fmt(ts.get('avg_prompt_tokens'),15,0)}")

    print()
    if g_ms and t_ms and g_ms > t_ms:
        ratio = g_ms / t_ms
        print(f"  >> grout prefill is {ratio:.2f}x SLOWER "
              f"({g_ms:.2f}ms vs {t_ms:.2f}ms, "
              f"{g_tps:.0f} vs {t_tps:.0f} tok/s).")
    elif g_ms and t_ms and t_ms > 0:
        print(f"  >> grout prefill is {t_ms/g_ms:.2f}x faster "
              f"({g_ms:.2f}ms vs {t_ms:.2f}ms).")
    else:
        print("  >> could not compute ratio (missing data).")


def bottleneck_analysis(g: dict, t: dict, top: int = 8):
    section("3. Bottleneck analysis (grout prefill)", "-")
    ops = g.get("op_profile") or []
    if not ops:
        print("  No op_profile in grout JSON.")
        print("  Re-run grout with: GROUT_PROFILE_OPS=1 --profile --max-new-tokens 0")
        print("  (run_comparison.sh does this automatically).")
        return

    # The op_profile comes from one run and lists per-op totals for that run.
    # Aggregate by op name just in case of duplicates.
    agg: dict[str, dict] = {}
    for o in ops:
        name = o["op"]
        a = agg.setdefault(name, {"total_ms": 0.0, "calls": 0, "avg_us": 0.0})
        a["total_ms"] += float(o.get("total_ms", 0.0))
        a["calls"] += int(o.get("calls", 0))
    for a in agg.values():
        a["avg_us"] = (a["total_ms"] * 1e3 / a["calls"]) if a["calls"] else 0.0

    ranked = sorted(agg.items(), key=lambda kv: kv[1]["total_ms"], reverse=True)

    # Reference: grout prefill wall-time (last measured run), in ms.
    runs = g.get("runs", [])
    prefill_ms = float(runs[-1].get("prompt_s", 0.0)) * 1000.0 if runs else None
    total_op_ms = sum(a["total_ms"] for _, a in ranked)
    # NOTE: op_profile timings are HOST-SIDE launch durations (CPU wall clock
    # between kernel dispatches), NOT GPU execution time. They do NOT include
    # the actual time kernels spend running on the GPU. So sum(op_ms) is almost
    # always << prefill_ms, and the gap = unaccounted GPU compute time.
    denom = prefill_ms or total_op_ms or 1.0
    unaccounted_ms = (prefill_ms or 0.0) - total_op_ms if prefill_ms else 0.0
    unaccounted_share = unaccounted_ms / denom * 100.0 if prefill_ms else 0.0

    if prefill_ms is not None:
        print(f"  grout prefill (last run):  {prefill_ms:.2f} ms")
    print(f"  sum of op launches:        {total_op_ms:.2f} ms "
          f"({total_op_ms/denom*100:.0f}% of prefill)")
    if prefill_ms and unaccounted_ms > 0:
        print(f"  UNACCOUNTED GPU time:      {unaccounted_ms:.2f} ms "
              f"({unaccounted_share:.0f}% of prefill)")
        print(f"    ^ This is real GPU compute not captured by host-side launch")
        print(f"      timers. It is where prefill actually spends its time.")
    print()
    print(f"  Per-op HOST-SIDE launch time (ranking, NOT GPU time):")
    print(f"  {'op':<16}{'launch_ms':>11}{'share':>9}{'calls':>8}{'avg_us':>11}")
    print(f"  {'-'*16}{'-'*11}{'-'*9}{'-'*8}{'-'*11}")
    for name, a in ranked[:top]:
        share = a["total_ms"] / denom * 100.0
        print(f"  {name:<16}{a['total_ms']:>11.3f}{share:>8.1f}%"
              f"{a['calls']:>8}{a['avg_us']:>11.2f}")
    if len(ranked) > top:
        print(f"  ... +{len(ranked)-top} more ops")

    # The real diagnosis: where is the unaccounted GPU time?
    truns = t.get("runs", [])
    if runs and truns and unaccounted_ms > 0:
        t_pm = float(truns[-1]["prompt_s"]) * 1000.0
        g_pm = float(runs[-1]["prompt_s"]) * 1000.0
        print()
        if t_pm > 0 and g_pm > t_pm:
            gap = g_pm - t_pm
            print(f"  >> DIAGNOSIS: grout spends {unaccounted_ms:.0f}ms in GPU")
            print(f"     execution that host timers can't see. vs transformers")
            print(f"     ({g_pm:.0f}ms vs {t_pm:.0f}ms), the gap is {gap:.0f}ms.")
            print(f"     Since the per-op launch times are tiny (~{total_op_ms:.1f}ms")
            print(f"     total), the slowdown is in actual KERNEL EXECUTION, not")
            print(f"     launch overhead. The most likely culprits:")
            print()
            print(f"       - GEMM kernels (MatMul): cuBLAS calls for q/k/v/o + gate/up")
            print(f"         projections. Compare against transformers' cuBLAS/cuBLASLt.")
            print(f"           -> try GROUT_CUBLAS_COMPUTE16=1, GROUT_CUBLAS_FAST_ALGO=1")
            print(f"       - Attention kernel: flash-attn tiling (ATTN_BN_PREFILL).")
            print(f"           -> try GROUT_ATTN_BN_PREFILL=16|32|64")
            print(f"       - Kernel JIT/compile: cutile compiles kernels lazily. The FIRST")
            print(f"         warmup run is much slower (pays compile cost), but steady-state")
            print(f"         prefill is still slow -> this is NOT the bottleneck after warmup.")
            print(f"       - To get TRUE GPU time per kernel, profile with nsys/ncu:")
            print(f"           nsys profile -t cuda ./target/release/grout ...")
    elif ranked:
        # Short prompts: launch overhead dominates, op ranking is meaningful.
        top_name, top_a = ranked[0]
        share = top_a["total_ms"] / denom * 100.0
        print()
        print(f"  >> Dominant op (by host launch time): {top_name} = "
              f"{top_a['total_ms']:.2f} ms ({share:.1f}% of prefill).")
        hints = {
            "MatMul": ("prefill GEMM", "GROUT_CUBLAS_COMPUTE16=1, GROUT_CUBLAS_FAST_ALGO=1"),
            "Attention": ("flash-attn kernel", "GROUT_ATTN_BN_PREFILL (try 16/32/64)"),
            "RmsNorm": ("RMSNorm kernels", "reduce per-layer kernel count"),
        }
        desc, hint = hints.get(top_name, ("see kernel", "GROUT_PROFILE_OPS for detail"))
        print(f"     what:   {desc}")
        print(f"     try:    {hint}")


def tuning_hints(g: dict):
    section("4. Grout tuning knobs to try", "-")
    print("  Environment variables (set before running grout):")
    knobs = [
        ("GROUT_CUBLAS_COMPUTE16=1",  "use FP16 accumulation in cuBLAS (faster, less precise)"),
        ("GROUT_CUBLAS_COMPUTE16_MAX_M=<n>", "cap M for FP16 compute"),
        ("GROUT_CUBLAS_FAST_ALGO=1",  "let cuBLAS pick a fast algorithm"),
        ("GROUT_ATTN_BN_PREFILL=<16|32|64>", "attention block-N for prefill tiling"),
        ("GROUT_PROFILE_OPS=1",       "emit per-op timings (what produced section 3)"),
    ]
    for k, desc in knobs:
        print(f"    {k:<34} {desc}")
    cfg = g.get("config", {})
    print()
    print("  Current state:")
    print(f"    cublas_compute16  = {cfg.get('cublas_compute16')}")
    print(f"    cuda_graph_decode = {cfg.get('cuda_graph_decode')}  "
          f"(irrelevant for prefill-only)")


if __name__ == "__main__":
    args = parse_args()
    g = load(args.grout)
    t = load(args.transformers)
    print(hr())
    print(" grout vs transformers — PREFILL comparison")
    print(f" grout:        {args.grout}  (engine={g.get('engine')})")
    print(f" transformers: {args.transformers}  (engine={t.get('engine')})")
    print(hr())
    check_config(g, t)
    prefill_comparison(g, t)
    bottleneck_analysis(g, t, top=args.top)
    tuning_hints(g)
    print()
    print(hr())
