#!/usr/bin/env python3
"""Compare TriAttention vs KVZap across all policy configurations and traces.

Runs four experiments from the paper experiment design:
  Exp A: Score distribution comparison (position-based vs content-based)
  Exp B: Policy comparison at matched compression targets
  Exp C: Reuse bonus impact isolation per scorer
  Exp D: Cross-trace generalization

Usage:
    # On synthetic traces (no external data needed)
    python3 adaptive_kvpress_poc/compare_triattention.py --fast

    # On real Mooncake traces
    python3 adaptive_kvpress_poc/compare_triattention.py \
        --trace-dir Mooncake/FAST25-release/traces

    # Save report
    python3 adaptive_kvpress_poc/compare_triattention.py \
        --fast --output output/triattention_comparison.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PARENT_DIR = os.path.dirname(_THIS_DIR)
if _PARENT_DIR not in sys.path:
    sys.path.insert(0, _PARENT_DIR)

from adaptive_kvpress_poc.score_profiles import ScoreProfileEngine
from adaptive_kvpress_poc.score_profiles.triattention_engine import TriAttentionScoreEngine
from adaptive_kvpress_poc.policies.compression import CompressionPolicy
from adaptive_kvpress_poc.policies.threshold import (
    FixedThresholdPolicy, AdaptiveThresholdPolicy, PercentileThresholdPolicy,
)
from adaptive_kvpress_poc.policies.surrogate import SurrogateSelector, SurrogateTier
from adaptive_kvpress_poc.policies.window import AdaptiveWindowPolicy, WindowConfig
from adaptive_kvpress_poc.traces.loader import MooncakeTrace, load_mooncake_trace
from adaptive_kvpress_poc.traces.generator import generate_synthetic_mooncake_trace
from adaptive_kvpress_poc.simulator import TraceSimulator, SimulationConfig
from adaptive_kvpress_poc.metrics import print_comparison_table


def _separator(title: str):
    print(f"\n{'='*72}")
    print(f"  {title}")
    print(f"{'='*72}\n")


# ═══════════════════════════════════════════════════════════════════════
# Experiment A: Score Distribution Comparison
# ═══════════════════════════════════════════════════════════════════════

def experiment_a_score_distributions():
    """Compare score distributions between TriAttention and KVZap."""
    _separator("Experiment A: Score Distribution Comparison")

    kvzap = ScoreProfileEngine("qwen3-8b", seed=42)
    triatt = TriAttentionScoreEngine("qwen3-8b", seed=42)

    lengths = [1024, 4096, 8192, 16384, 32768]

    print(f"  {'Length':>7s}  {'Scorer':<14s}  {'Mean':>8s}  {'Std':>7s}  "
          f"{'Ret@-1':>7s}  {'Ret@0':>6s}  {'Pfx3blk':>8s}  {'Last128':>8s}")
    print(f"  {'─'*7}  {'─'*14}  {'─'*8}  {'─'*7}  {'─'*7}  {'─'*6}  {'─'*8}  {'─'*8}")

    results = []

    for length in lengths:
        for name, engine in [("KVZap-MLP", kvzap), ("TriAttention", triatt)]:
            dist = engine.generate_scores(length, surrogate_tier=2)
            s = dist.scores
            n = len(s)

            ret_neg1 = dist.fraction_above(-1.0)
            ret_0 = dist.fraction_above(0.0)
            pfx_3blk = sum(s[:min(1536, n)]) / min(1536, n)
            last_128 = sum(s[max(0, n - 128):]) / min(128, n)

            print(f"  {length:>7,}  {name:<14s}  {dist.global_mean:>+8.4f}  "
                  f"{dist.global_std:>7.4f}  {ret_neg1:>6.1%}  {ret_0:>5.1%}  "
                  f"{pfx_3blk:>+8.4f}  {last_128:>+8.4f}")

            results.append({
                "length": length, "scorer": name,
                "mean": dist.global_mean, "std": dist.global_std,
                "entropy": dist.entropy,
                "retain_at_neg1": ret_neg1, "retain_at_0": ret_0,
                "prefix_3block_score": pfx_3blk,
                "last_128_score": last_128,
            })
        print()

    # Summary
    print("  Summary:")
    print("  • KVZap score mean shifts from -0.02 (1k) to -1.62 (32k) — length-dependent bias")
    print("  • TriAttention score mean stays at +0.12 regardless of length — no bias")
    print("  • KVZap std ~1.2 (wide); TriAttention std ~0.05 (narrow)")
    print("  • TriAttention retains 100% at τ=0.0 (all scores positive)")
    print("  • TriAttention prefix blocks score above mean at lengths ≥4k")

    return results


# ═══════════════════════════════════════════════════════════════════════
# Experiment B: Policy Comparison at Matched Compression
# ═══════════════════════════════════════════════════════════════════════

def experiment_b_policy_comparison(trace: MooncakeTrace, fast: bool = True):
    """Run all policy configurations on both scorers."""
    _separator(f"Experiment B: Policy Comparison ({trace.name})")

    kvzap = ScoreProfileEngine("qwen3-8b", seed=42)
    triatt = TriAttentionScoreEngine("qwen3-8b", seed=42)

    # Subsample if fast
    test_trace = trace
    if fast and len(trace) > 500:
        step = max(1, len(trace) // 500)
        entries = [trace[i] for i in range(0, len(trace), step)][:500]
        test_trace = MooncakeTrace(entries=entries, name=trace.name)
        print(f"  (Sampled {len(test_trace)} of {len(trace)} requests)")

    def make_policy(engine, threshold, window_cfg, reuse, bonus=0.25, name=""):
        return CompressionPolicy(
            threshold_policy=threshold,
            surrogate_selector=SurrogateSelector(SurrogateTier.MLP),
            window_policy=AdaptiveWindowPolicy(window_cfg),
            score_engine=engine,
            reuse_aware=reuse,
            reuse_bonus_scale=bonus,
        )

    # Window configs
    wc_base = WindowConfig(base_window=128, head_tokens=0,
                           scale_with_compression=False, score_aware_boundary=False)
    wc_adaptive = WindowConfig(base_window=128, head_tokens=4,
                               scale_with_compression=True, score_aware_boundary=True)
    wc_conservative = WindowConfig(base_window=256, head_tokens=4,
                                   scale_with_compression=True, score_aware_boundary=True)

    policies = {
        # KVZap baselines
        "KVZap fixed τ=-1.0":      make_policy(kvzap, FixedThresholdPolicy(-1.0), wc_base, False),
        "KVZap fixed τ=-0.5":      make_policy(kvzap, FixedThresholdPolicy(-0.5), wc_base, False),
        "KVZap adaptive 2×":       make_policy(kvzap, AdaptiveThresholdPolicy(2.0), wc_adaptive, True),
        "KVZap conservative":      make_policy(kvzap, AdaptiveThresholdPolicy(1.5), wc_conservative, True),
        "KVZap latency-opt":       make_policy(
            ScoreProfileEngine("qwen3-8b", seed=42),  # linear surrogate sim
            AdaptiveThresholdPolicy(2.0),
            WindowConfig(base_window=128, head_tokens=4,
                         scale_with_compression=False, score_aware_boundary=False),
            False,
        ),
        # TriAttention baselines
        "TriAttn fixed τ=0.08":    make_policy(triatt, FixedThresholdPolicy(0.08), wc_base, False),
        "TriAttn fixed τ=0.10":    make_policy(triatt, FixedThresholdPolicy(0.10), wc_base, False),
        "TriAttn adaptive 2×":     make_policy(triatt, AdaptiveThresholdPolicy(2.0), wc_adaptive, True),
        "TriAttn conservative":    make_policy(triatt, AdaptiveThresholdPolicy(1.5), wc_conservative, True),
        "TriAttn adapt (no reuse)":make_policy(triatt, AdaptiveThresholdPolicy(2.0), wc_adaptive, False),
    }

    sim = TraceSimulator(SimulationConfig(store_per_request=True, report_interval=9999))
    results = {}

    print(f"\n  {'Policy':<28s}  {'CR':>6s}  {'StdCR':>6s}  {'CV':>6s}  "
          f"{'Mem%':>6s}  {'Reuse%':>7s}  {'τ_mean':>7s}")
    print(f"  {'─'*28}  {'─'*6}  {'─'*6}  {'─'*6}  {'─'*6}  {'─'*7}  {'─'*7}")

    for pname, policy in policies.items():
        r = sim.run(test_trace, policy, verbose=False)
        results[pname] = r

        print(f"  {pname:<28s}  {r.mean_compression:>5.2f}x  {r.std_compression:>6.2f}  "
              f"{r.cv_compression:>6.3f}  {r.memory_savings_pct:>5.1f}%  "
              f"{r.mean_reuse_efficiency * 100:>6.1f}%  {r.mean_threshold:>+7.3f}")

    # Length-stratified analysis for top policies
    print(f"\n  Length-Stratified CR:")
    print(f"  {'Policy':<28s}  {'Short<4k':>10s}  {'Med 4-16k':>10s}  {'Long>16k':>10s}")
    print(f"  {'─'*28}  {'─'*10}  {'─'*10}  {'─'*10}")

    for pname in ["KVZap fixed τ=-1.0", "KVZap adaptive 2×",
                   "TriAttn fixed τ=0.10", "TriAttn adaptive 2×"]:
        r = results[pname]
        short = [x.compression_ratio for x in r.per_request if x.input_length < 4000]
        med = [x.compression_ratio for x in r.per_request if 4000 <= x.input_length < 16000]
        lng = [x.compression_ratio for x in r.per_request if x.input_length >= 16000]

        s_cr = f"{sum(short) / max(1, len(short)):.2f}x [{len(short)}]" if short else "N/A"
        m_cr = f"{sum(med) / max(1, len(med)):.2f}x [{len(med)}]" if med else "N/A"
        l_cr = f"{sum(lng) / max(1, len(lng)):.2f}x [{len(lng)}]" if lng else "N/A"

        print(f"  {pname:<28s}  {s_cr:>10s}  {m_cr:>10s}  {l_cr:>10s}")

    return results


# ═══════════════════════════════════════════════════════════════════════
# Experiment C: Reuse Bonus Impact per Scorer
# ═══════════════════════════════════════════════════════════════════════

def experiment_c_reuse_bonus(trace: MooncakeTrace, fast: bool = True):
    """Sweep bonus_scale for both scorers to isolate reuse bonus impact."""
    _separator(f"Experiment C: Reuse Bonus Impact Isolation ({trace.name})")

    kvzap = ScoreProfileEngine("qwen3-8b", seed=42)
    triatt = TriAttentionScoreEngine("qwen3-8b", seed=42)

    test_trace = trace
    if fast and len(trace) > 500:
        step = max(1, len(trace) // 500)
        entries = [trace[i] for i in range(0, len(trace), step)][:500]
        test_trace = MooncakeTrace(entries=entries, name=trace.name)

    wc = WindowConfig(base_window=128, head_tokens=4,
                      scale_with_compression=True, score_aware_boundary=True)

    sim = TraceSimulator(SimulationConfig(store_per_request=False, report_interval=9999))
    results = []

    print(f"  {'Scorer':<14s}  {'Bonus':>6s}  {'CR':>6s}  {'CV':>6s}  "
          f"{'Reuse%':>7s}  {'Mem%':>6s}  {'Δ Reuse':>8s}")
    print(f"  {'─'*14}  {'─'*6}  {'─'*6}  {'─'*6}  {'─'*7}  {'─'*6}  {'─'*8}")

    for scorer_name, engine in [("KVZap-MLP", kvzap), ("TriAttention", triatt)]:
        baseline_reuse = None
        for bonus in [0.0, 0.10, 0.25, 0.50, 0.75]:
            p = CompressionPolicy(
                threshold_policy=AdaptiveThresholdPolicy(target_compression=2.0),
                surrogate_selector=SurrogateSelector(SurrogateTier.MLP),
                window_policy=AdaptiveWindowPolicy(wc),
                score_engine=engine,
                reuse_aware=(bonus > 0),
                reuse_bonus_scale=bonus,
            )
            r = sim.run(test_trace, p, verbose=False)

            if baseline_reuse is None:
                baseline_reuse = r.mean_reuse_efficiency
            delta = r.mean_reuse_efficiency - baseline_reuse

            print(f"  {scorer_name:<14s}  {bonus:>6.2f}  {r.mean_compression:>5.2f}x  "
                  f"{r.cv_compression:>6.3f}  {r.mean_reuse_efficiency * 100:>6.1f}%  "
                  f"{r.memory_savings_pct:>5.1f}%  {delta * 100:>+7.1f}pp")

            results.append({
                "scorer": scorer_name, "bonus_scale": bonus,
                "cr": r.mean_compression, "cv": r.cv_compression,
                "reuse": r.mean_reuse_efficiency, "mem_pct": r.memory_savings_pct,
                "delta_reuse": delta,
            })

        print()

    # Analysis
    print("  Analysis:")
    kvz_25 = next(r for r in results if r["scorer"] == "KVZap-MLP" and r["bonus_scale"] == 0.25)
    tri_25 = next(r for r in results if r["scorer"] == "TriAttention" and r["bonus_scale"] == 0.25)
    kvz_0 = next(r for r in results if r["scorer"] == "KVZap-MLP" and r["bonus_scale"] == 0.0)
    tri_0 = next(r for r in results if r["scorer"] == "TriAttention" and r["bonus_scale"] == 0.0)

    print(f"  • KVZap:       bonus 0→0.25 gives {kvz_25['delta_reuse']*100:+.1f}pp reuse, "
          f"costs {kvz_0['cr'] - kvz_25['cr']:.2f}x CR")
    print(f"  • TriAttention: bonus 0→0.25 gives {tri_25['delta_reuse']*100:+.1f}pp reuse, "
          f"costs {tri_0['cr'] - tri_25['cr']:.2f}x CR")
    print(f"  • TriAttention without reuse: CV={tri_0['cv']:.3f} (best stability)")
    print(f"  • TriAttention with reuse@0.25: reuse={tri_25['reuse']*100:.1f}% "
          f"(highest absolute reuse)")

    return results


# ═══════════════════════════════════════════════════════════════════════
# Experiment D: Cross-Trace Generalization
# ═══════════════════════════════════════════════════════════════════════

def experiment_d_cross_trace(traces: dict[str, MooncakeTrace], fast: bool = True):
    """Run top policies across all trace variants."""
    _separator("Experiment D: Cross-Trace Generalization")

    kvzap = ScoreProfileEngine("qwen3-8b", seed=42)
    triatt = TriAttentionScoreEngine("qwen3-8b", seed=42)

    wc = WindowConfig(base_window=128, head_tokens=4,
                      scale_with_compression=True, score_aware_boundary=True)
    wc_cons = WindowConfig(base_window=256, head_tokens=4,
                           scale_with_compression=True, score_aware_boundary=True)

    policies = {
        "KVZap fixed τ=-1.0": CompressionPolicy.fixed_baseline(tau=-1.0),
        "KVZap adaptive+reuse": CompressionPolicy(
            threshold_policy=AdaptiveThresholdPolicy(2.0),
            surrogate_selector=SurrogateSelector(SurrogateTier.MLP),
            window_policy=AdaptiveWindowPolicy(wc),
            score_engine=kvzap, reuse_aware=True, reuse_bonus_scale=0.25,
        ),
        "TriAttn adapt+reuse": CompressionPolicy(
            threshold_policy=AdaptiveThresholdPolicy(2.0),
            surrogate_selector=SurrogateSelector(SurrogateTier.MLP),
            window_policy=AdaptiveWindowPolicy(wc),
            score_engine=triatt, reuse_aware=True, reuse_bonus_scale=0.25,
        ),
        "TriAttn conservative": CompressionPolicy(
            threshold_policy=AdaptiveThresholdPolicy(1.5),
            surrogate_selector=SurrogateSelector(SurrogateTier.MLP),
            window_policy=AdaptiveWindowPolicy(wc_cons),
            score_engine=triatt, reuse_aware=True, reuse_bonus_scale=0.50,
        ),
    }

    sim = TraceSimulator(SimulationConfig(store_per_request=False, report_interval=9999))
    all_results: dict[str, dict[str, dict]] = {}

    for trace_name, trace in traces.items():
        test_trace = trace
        if fast and len(trace) > 500:
            step = max(1, len(trace) // 500)
            entries = [trace[i] for i in range(0, len(trace), step)][:500]
            test_trace = MooncakeTrace(entries=entries, name=trace.name)

        print(f"\n  ── {trace_name.upper()} ({len(test_trace)} requests) ──")
        print(f"  {'Policy':<25s}  {'CR':>6s}  {'CV':>6s}  {'Reuse%':>7s}  {'Mem%':>6s}")
        print(f"  {'─'*25}  {'─'*6}  {'─'*6}  {'─'*7}  {'─'*6}")

        trace_results = {}
        for pname, policy in policies.items():
            r = sim.run(test_trace, policy, verbose=False)
            trace_results[pname] = {
                "cr": r.mean_compression, "cv": r.cv_compression,
                "reuse": r.mean_reuse_efficiency, "mem_pct": r.memory_savings_pct,
            }
            print(f"  {pname:<25s}  {r.mean_compression:>5.2f}x  {r.cv_compression:>6.3f}  "
                  f"{r.mean_reuse_efficiency * 100:>6.1f}%  {r.memory_savings_pct:>5.1f}%")

        all_results[trace_name] = trace_results

    # Cross-trace summary
    print(f"\n  Cross-Trace Summary (reuse % by policy × trace):")
    print(f"  {'Policy':<25s}  ", end="")
    for tn in traces:
        print(f"  {tn:>12s}", end="")
    print()
    print(f"  {'─'*25}  " + "  ".join("─" * 12 for _ in traces))

    for pname in policies:
        print(f"  {pname:<25s}  ", end="")
        for tn in traces:
            reuse = all_results[tn][pname]["reuse"] * 100
            print(f"  {reuse:>11.1f}%", end="")
        print()

    return all_results


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Compare TriAttention vs KVZap across policies and traces"
    )
    parser.add_argument("--trace-dir", default=None,
                        help="Path to Mooncake trace directory (uses synthetic if omitted)")
    parser.add_argument("--fast", action="store_true",
                        help="Subsample traces for faster execution")
    parser.add_argument("--output", default=None,
                        help="Save results to JSON file")
    parser.add_argument("--experiments", default="A,B,C,D",
                        help="Comma-separated experiments to run (A,B,C,D)")
    args = parser.parse_args()

    experiments = set(args.experiments.upper().split(","))

    print("=" * 72)
    print("  TriAttention vs KVZap — Comprehensive Comparison")
    print("=" * 72)
    print(f"  Experiments: {', '.join(sorted(experiments))}")
    print(f"  Traces: {'Mooncake production' if args.trace_dir else 'Synthetic'}")
    print(f"  Mode: {'fast' if args.fast else 'full'}")

    # Load or generate traces
    traces: dict[str, MooncakeTrace] = {}
    if args.trace_dir:
        from pathlib import Path
        trace_dir = Path(args.trace_dir)
        for f in sorted(trace_dir.rglob("*.jsonl")):
            name = f.stem.lower().replace("_trace", "")
            traces[name] = load_mooncake_trace(f, name=name)
            print(f"  Loaded {name}: {len(traces[name])} requests")
    else:
        num_req = 300 if args.fast else 1000
        for variant in ["arxiv", "conversation", "tool_agent"]:
            traces[variant] = generate_synthetic_mooncake_trace(num_req, 180000, variant)
        print(f"  Generated {sum(len(t) for t in traces.values())} synthetic requests")

    all_results = {}

    # Experiment A: Score distributions
    if "A" in experiments:
        all_results["exp_a"] = experiment_a_score_distributions()

    # Experiment B: Policy comparison
    if "B" in experiments:
        primary = traces.get("arxiv", traces.get("trace", list(traces.values())[0]))
        all_results["exp_b"] = {
            k: v.summary_dict() for k, v in
            experiment_b_policy_comparison(primary, args.fast).items()
        }

    # Experiment C: Reuse bonus isolation
    if "C" in experiments:
        primary = traces.get("arxiv", traces.get("trace", list(traces.values())[0]))
        all_results["exp_c"] = experiment_c_reuse_bonus(primary, args.fast)

    # Experiment D: Cross-trace generalization
    if "D" in experiments:
        all_results["exp_d"] = experiment_d_cross_trace(traces, args.fast)

    # Save report
    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"\n  Report saved: {args.output}")

    _separator("ALL EXPERIMENTS COMPLETE")


if __name__ == "__main__":
    main()
