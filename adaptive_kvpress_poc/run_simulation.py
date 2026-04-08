#!/usr/bin/env python3
"""Run trace-driven adaptive KV cache compression simulation.

This is the main entry point for the PoC. It:
  1. Generates synthetic Mooncake traces (arxiv, conversation, tool_agent)
  2. Defines multiple compression policies (baseline + adaptive variants)
  3. Runs trace replay simulation for each policy × trace combination
  4. Produces comparison reports and JSON data for visualization
"""

import json
import os
import sys
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from adaptive_kvpress_poc.traces import (
    generate_synthetic_mooncake_trace,
    generate_synthetic_azure_trace,
)
from adaptive_kvpress_poc.traces.generator import save_trace_jsonl
from adaptive_kvpress_poc.policies import CompressionPolicy
from adaptive_kvpress_poc.simulator import (
    TraceSimulator,
    SimulationConfig,
    run_comparison,
)
from adaptive_kvpress_poc.metrics import (
    generate_comparison_report,
    print_comparison_table,
)


def main(fast: bool = False):
    num_requests = 500 if fast else 2000
    duration_ms = 120_000 if fast else 300_000

    print("=" * 70)
    print("  Adaptive KVPress PoC — Trace-Driven Compression Policy Simulator")
    print("=" * 70)
    print()

    # ── 1. Generate synthetic traces ─────────────────────────────────────

    print("▸ Generating synthetic Mooncake traces...")

    traces = {}
    for variant in ["arxiv", "conversation", "tool_agent"]:
        t = generate_synthetic_mooncake_trace(
            num_requests=num_requests,
            duration_ms=duration_ms,
            variant=variant,
        )
        traces[variant] = t
        summary = t.summary()
        print(f"  {variant}: {summary['num_requests']} requests, "
              f"mean_input={summary['mean_input_tokens']:.0f} tokens, "
              f"p99_input={summary['p99_input']} tokens, "
              f"unique_blocks={summary['unique_blocks']}")

    # Also generate Azure trace for cross-provider comparison
    azure_trace = generate_synthetic_azure_trace(
        num_requests=num_requests // 2,
        duration_ms=duration_ms,
    )
    traces["azure"] = azure_trace
    print(f"  azure: {len(azure_trace)} requests, "
          f"mean_input={azure_trace.mean_input_length:.0f} tokens")

    # Save traces to JSONL (for reproducibility)
    os.makedirs("output/traces", exist_ok=True)
    for name, trace in traces.items():
        save_trace_jsonl(trace, f"output/traces/{name}.jsonl")
    print(f"\n  Traces saved to output/traces/")

    # ── 2. Define compression policies ───────────────────────────────────

    print("\n▸ Configuring compression policies...")

    policies = {
        # Baselines
        "fixed_tau=-1.0": CompressionPolicy.fixed_baseline(tau=-1.0),
        "fixed_tau=-0.5": CompressionPolicy.fixed_baseline(tau=-0.5),
        "fixed_tau=0.0": CompressionPolicy.fixed_baseline(tau=0.0),
        "percentile_50%": CompressionPolicy.percentile_baseline(prune_fraction=0.50),

        # Adaptive variants
        "adaptive_2x": CompressionPolicy.balanced(target_compression=2.0),
        "adaptive_3x": CompressionPolicy.balanced(target_compression=3.0),
        "conservative": CompressionPolicy.conservative(),
        "aggressive": CompressionPolicy.aggressive(),
        "latency_optimal": CompressionPolicy.latency_optimal(),
    }

    for name, policy in policies.items():
        print(f"  {name:25s} → {policy.name[:60]}")

    # ── 3. Run simulations ───────────────────────────────────────────────

    all_results = {}
    config = SimulationConfig(store_per_request=True, report_interval=500)

    # Primary comparison: all policies on arxiv trace
    print("\n" + "=" * 70)
    print("  Phase 1: Policy comparison on arxiv trace")
    print("=" * 70)

    arxiv_results = run_comparison(
        traces["arxiv"], policies, config, verbose=not fast
    )
    all_results["arxiv"] = arxiv_results

    print("\n\n▸ Comparison Table (arxiv trace):")
    print_comparison_table(arxiv_results)

    # Cross-trace: balanced policy on all trace variants
    print("\n" + "=" * 70)
    print("  Phase 2: Cross-trace generalization (adaptive_2x policy)")
    print("=" * 70)

    balanced_policy = policies["adaptive_2x"]
    cross_trace_results = {}
    simulator = TraceSimulator(config)

    for trace_name, trace in traces.items():
        print(f"\n  Running on {trace_name} trace...")
        result = simulator.run(trace, balanced_policy, verbose=False)
        cross_trace_results[trace_name] = result
        print(f"    mean_cr={result.mean_compression:.2f}x "
              f"cv={result.cv_compression:.3f} "
              f"reuse_eff={result.mean_reuse_efficiency:.3f}")

    all_results["cross_trace"] = cross_trace_results

    print("\n\n▸ Cross-Trace Comparison (adaptive_2x):")
    print_comparison_table(cross_trace_results)

    # ── 4. Ablation: window size sweep ───────────────────────────────────

    print("\n" + "=" * 70)
    print("  Phase 3: Window size ablation")
    print("=" * 70)

    from adaptive_kvpress_poc.policies.window import AdaptiveWindowPolicy, WindowConfig
    from adaptive_kvpress_poc.policies.threshold import AdaptiveThresholdPolicy
    from adaptive_kvpress_poc.policies.surrogate import SurrogateSelector, SurrogateTier
    from adaptive_kvpress_poc.score_profiles import ScoreProfileEngine

    window_results = {}
    for window_size in [64, 128, 192, 256, 384, 512]:
        for sinks in [0, 4]:
            name = f"w={window_size},sinks={sinks}"
            wp = AdaptiveWindowPolicy(WindowConfig(
                base_window=window_size,
                head_tokens=sinks,
                scale_with_compression=True,
                score_aware_boundary=(sinks > 0),
            ))
            p = CompressionPolicy(
                threshold_policy=AdaptiveThresholdPolicy(target_compression=2.5),
                surrogate_selector=SurrogateSelector(SurrogateTier.MLP),
                window_policy=wp,
                score_engine=ScoreProfileEngine("qwen3-8b"),
                reuse_aware=(sinks > 0),
            )
            result = simulator.run(traces["arxiv"], p, verbose=False)
            window_results[name] = result
            print(f"  {name:25s} → cr={result.mean_compression:.2f}x "
                  f"reuse={result.mean_reuse_efficiency:.3f}")

    all_results["window_ablation"] = window_results

    # ── 5. Generate reports ──────────────────────────────────────────────

    print("\n" + "=" * 70)
    print("  Generating reports...")
    print("=" * 70)

    os.makedirs("output/reports", exist_ok=True)

    # Main comparison report
    report = generate_comparison_report(
        arxiv_results,
        output_path="output/reports/arxiv_comparison.json",
    )
    print(f"  arxiv_comparison.json: {len(report['summary_table'])} policies")

    # Cross-trace report
    cross_report = generate_comparison_report(
        cross_trace_results,
        output_path="output/reports/cross_trace.json",
    )
    print(f"  cross_trace.json: {len(cross_report['summary_table'])} traces")

    # Window ablation report
    window_report = generate_comparison_report(
        window_results,
        output_path="output/reports/window_ablation.json",
    )
    print(f"  window_ablation.json: {len(window_report['summary_table'])} configs")

    # Combined dashboard data
    dashboard_data = {
        "arxiv_comparison": report,
        "cross_trace": cross_report,
        "window_ablation": window_report,
        "trace_summaries": {name: t.summary() for name, t in traces.items()},
    }
    with open("output/reports/dashboard_data.json", "w") as f:
        json.dump(dashboard_data, f, indent=2)
    print(f"  dashboard_data.json: combined data for visualization")

    print("\n" + "=" * 70)
    print("  Simulation complete!")
    print("=" * 70)
    print(f"\n  Output files in output/")
    print(f"    traces/     - Mooncake-format JSONL trace files")
    print(f"    reports/    - JSON comparison reports")

    return dashboard_data


if __name__ == "__main__":
    fast = "--fast" in sys.argv
    main(fast=fast)
