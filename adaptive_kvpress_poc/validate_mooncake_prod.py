#!/usr/bin/env python3
"""Validate adaptive compression PoC against real Mooncake production traces.
 
Usage:
    # Step 1: Download traces (see instructions below)
    # Step 2: Run validation
    python3 adaptive_kvpress_poc/validate_mooncake.py \
        --trace-dir Mooncake/FAST25-release/traces \
        --fast
 
This script:
  1. Loads real Mooncake production traces (JSONL format)
  2. Compares trace statistics against synthetic generator output
  3. Runs all 7 experiments from deep_analysis.py on real data
  4. Produces a validation report comparing synthetic vs production results
"""
 
from __future__ import annotations
 
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
 
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PARENT_DIR = os.path.dirname(_THIS_DIR)
if _PARENT_DIR not in sys.path:
    sys.path.insert(0, _PARENT_DIR)
 
from adaptive_kvpress_poc.traces.loader import MooncakeTrace, TraceEntry, load_mooncake_trace
from adaptive_kvpress_poc.traces.generator import generate_synthetic_mooncake_trace
from adaptive_kvpress_poc.policies.compression import CompressionPolicy
from adaptive_kvpress_poc.simulator import TraceSimulator, SimulationConfig
from adaptive_kvpress_poc.metrics import print_comparison_table, generate_comparison_report
 
 
# ═══════════════════════════════════════════════════════════════════════
# STEP 1: Locate and load traces
# ═══════════════════════════════════════════════════════════════════════
 
def find_trace_files(trace_dir: str) -> dict[str, Path]:
    """Discover all JSONL trace files in the Mooncake trace directory.
 
    Expected structure (FAST25-release/traces/):
      trace.jsonl                    ← original arxiv trace (July 2024, 23,608 req)
      conversation_trace.jsonl       ← FAST'25 updated (Feb 2025)
      tool_agent_trace.jsonl         ← FAST'25 updated
      synthetic_trace.jsonl          ← FAST'25 updated
 
    File names may vary. We detect by scanning for .jsonl files.
    """
    trace_dir = Path(trace_dir)
    if not trace_dir.exists():
        print(f"  ERROR: Trace directory not found: {trace_dir}")
        print(f"  Download instructions:")
        print(f"    git clone https://github.com/kvcache-ai/Mooncake.git")
        print(f"    # Traces are in Mooncake/FAST25-release/traces/")
        sys.exit(1)
 
    found = {}
    for f in sorted(trace_dir.rglob("*.jsonl")):
        name = f.stem.lower()
        # Classify by filename
        if "conversation" in name:
            found["conversation"] = f
        elif "tool" in name or "agent" in name:
            found["tool_agent"] = f
        elif "synthetic" in name:
            found["synthetic"] = f
        elif "trace" in name or "arxiv" in name:
            found["arxiv"] = f
        else:
            # Unknown variant — include with its stem name
            found[name] = f
 
    if not found:
        # Try parent directory or current directory
        for f in sorted(Path(trace_dir).glob("*.jsonl")):
            found[f.stem] = f
 
    return found
 
 
def validate_trace_format(path: Path) -> tuple[bool, str, int]:
    """Validate a JSONL file matches Mooncake's expected schema.
 
    Returns (is_valid, message, num_entries).
    """
    required_fields = {"timestamp", "input_length", "output_length"}
    optional_fields = {"hash_ids"}
 
    count = 0
    errors = []
 
    with open(path) as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                errors.append(f"  Line {i}: Invalid JSON: {e}")
                if len(errors) >= 5:
                    break
                continue
 
            missing = required_fields - set(obj.keys())
            if missing:
                errors.append(f"  Line {i}: Missing fields: {missing}")
                if len(errors) >= 5:
                    break
                continue
 
            count += 1
 
    if errors:
        return False, "\n".join(errors), count
    return True, f"Valid: {count} entries", count
 
 
# ═══════════════════════════════════════════════════════════════════════
# STEP 2: Statistical comparison — synthetic vs production
# ═══════════════════════════════════════════════════════════════════════
 
def compare_trace_stats(real: MooncakeTrace, synthetic: MooncakeTrace, name: str):
    """Print side-by-side comparison of trace statistics."""
 
    def stats(trace):
        inputs = sorted([e.input_length for e in trace])
        outputs = sorted([e.output_length for e in trace])
        n = len(inputs)
        bf = trace.block_frequency
        return {
            "n": n,
            "input_mean": sum(inputs) / n,
            "input_p50": inputs[n // 2],
            "input_p90": inputs[int(n * 0.9)],
            "input_p99": inputs[int(n * 0.99)],
            "input_max": inputs[-1],
            "output_mean": sum(outputs) / n,
            "unique_blocks": len(bf),
            "total_refs": sum(bf.values()) if bf else 0,
            "hot_blocks": sum(1 for f in bf.values() if f >= 10) if bf else 0,
            "cold_blocks": sum(1 for f in bf.values() if f == 1) if bf else 0,
        }
 
    r = stats(real)
    s = stats(synthetic)
 
    print(f"\n  ┌─ {name.upper()} ─ Synthetic vs Production Comparison ────────────┐")
    print(f"  │ {'Metric':<25s} {'Production':>12s} {'Synthetic':>12s} {'Δ%':>8s} │")
    print(f"  ├─{'─'*25}─{'─'*12}─{'─'*12}─{'─'*8}─┤")
 
    for label, rk, sk in [
        ("Requests",        "n",            "n"),
        ("Mean input tok",  "input_mean",   "input_mean"),
        ("P50 input",       "input_p50",    "input_p50"),
        ("P90 input",       "input_p90",    "input_p90"),
        ("P99 input",       "input_p99",    "input_p99"),
        ("Max input",       "input_max",    "input_max"),
        ("Mean output tok", "output_mean",  "output_mean"),
        ("Unique blocks",   "unique_blocks","unique_blocks"),
        ("Total block refs", "total_refs",  "total_refs"),
        ("Hot blocks (f>=10)", "hot_blocks","hot_blocks"),
        ("Cold blocks (f=1)","cold_blocks", "cold_blocks"),
    ]:
        rv, sv = r[rk], s[sk]
        if rv > 0:
            delta = f"{(sv - rv) / rv * 100:+.1f}%"
        else:
            delta = "N/A"
        print(f"  │ {label:<25s} {rv:>12,.0f} {sv:>12,.0f} {delta:>8s} │")
 
    print(f"  └─{'─'*61}─┘")
 
 
# ═══════════════════════════════════════════════════════════════════════
# STEP 3: Run full policy comparison on real trace
# ═══════════════════════════════════════════════════════════════════════
 
def run_policy_comparison(trace: MooncakeTrace, name: str, fast: bool = True):
    """Run all policies from deep_analysis.py on a real trace."""
 
    policies = {
        "fixed_tau=-1.0": CompressionPolicy.fixed_baseline(tau=-1.0),
        "fixed_tau=-0.5": CompressionPolicy.fixed_baseline(tau=-0.5),
        "percentile_50%": CompressionPolicy.percentile_baseline(0.5),
        "adaptive_2x": CompressionPolicy.balanced(target_compression=2.0),
        "adaptive_3x": CompressionPolicy.balanced(target_compression=3.0),
        "conservative": CompressionPolicy.conservative(),
        "latency_optimal": CompressionPolicy.latency_optimal(),
    }
 
    # If fast mode and trace is large, subsample
    test_trace = trace
    if fast and len(trace) > 2000:
        step = len(trace) // 2000
        entries = [trace[i] for i in range(0, len(trace), step)][:2000]
        test_trace = MooncakeTrace(entries=entries, name=f"{trace.name}_sampled")
        print(f"  (Fast mode: sampled {len(test_trace)} of {len(trace)} requests)")
 
    sim = TraceSimulator(SimulationConfig(store_per_request=True, report_interval=9999))
    results = {}
 
    for pname, policy in policies.items():
        result = sim.run(test_trace, policy, verbose=False)
        results[pname] = result
 
    print(f"\n  Policy Comparison on {name} (production trace):")
    print_comparison_table(results)
 
    return results
 
 
# ═══════════════════════════════════════════════════════════════════════
# STEP 4: Reuse bonus sweep on real trace
# ═══════════════════════════════════════════════════════════════════════
 
def run_reuse_bonus_sweep(trace: MooncakeTrace, name: str, fast: bool = True):
    """Run Experiment 4 (reuse bonus isolation) on real trace."""
    from adaptive_kvpress_poc.policies.threshold import AdaptiveThresholdPolicy
    from adaptive_kvpress_poc.policies.surrogate import SurrogateSelector, SurrogateTier
    from adaptive_kvpress_poc.policies.window import AdaptiveWindowPolicy, WindowConfig
    from adaptive_kvpress_poc.score_profiles import ScoreProfileEngine
 
    test_trace = trace
    if fast and len(trace) > 2000:
        step = len(trace) // 2000
        entries = [trace[i] for i in range(0, len(trace), step)][:2000]
        test_trace = MooncakeTrace(entries=entries, name=f"{trace.name}_sampled")
 
    sim = TraceSimulator(SimulationConfig(store_per_request=False, report_interval=9999))
 
    print(f"\n  Reuse Bonus Sweep on {name} (production trace):")
    print(f"  {'bonus_scale':<14s}  {'CR':>6s}  {'CV':>6s}  {'Reuse':>8s}  {'Mem Saved':>10s}")
    print(f"  {'─'*14}  {'─'*6}  {'─'*6}  {'─'*8}  {'─'*10}")
 
    for bonus_scale in [0.0, 0.10, 0.25, 0.50, 0.75]:
        p = CompressionPolicy(
            threshold_policy=AdaptiveThresholdPolicy(target_compression=2.0),
            surrogate_selector=SurrogateSelector(SurrogateTier.MLP),
            window_policy=AdaptiveWindowPolicy(WindowConfig(
                base_window=128, head_tokens=4,
                scale_with_compression=True, score_aware_boundary=True)),
            score_engine=ScoreProfileEngine("qwen3-8b"),
            reuse_aware=(bonus_scale > 0),
            reuse_bonus_scale=bonus_scale,
        )
        r = sim.run(test_trace, p, verbose=False)
        print(f"  {bonus_scale:<14.2f}  {r.mean_compression:>5.2f}x  {r.cv_compression:>6.3f}  "
              f"{r.mean_reuse_efficiency*100:>7.1f}%  {r.memory_savings_pct:>9.1f}%")
 
 
# ═══════════════════════════════════════════════════════════════════════
# STEP 5: Prefix reuse depth on real trace
# ═══════════════════════════════════════════════════════════════════════
 
def run_prefix_depth_analysis(trace: MooncakeTrace, name: str):
    """Run Experiment 7 (prefix reuse depth) on real trace."""
    print(f"\n  Prefix Reuse Depth on {name} (production trace):")
 
    if not any(e.hash_ids for e in trace):
        print(f"  SKIPPED: No hash_ids in trace entries.")
        return
 
    n_pairs = min(500, len(trace) - 1)
    overlaps = []
    for i in range(1, n_pairs + 1):
        overlap = trace[i].prefix_overlap(trace[i - 1])
        overlaps.append(overlap)
 
    if not overlaps:
        print(f"  No sequential pairs to analyze.")
        return
 
    overlaps_s = sorted(overlaps)
    n = len(overlaps_s)
    BLOCK_SIZE = 512
 
    token_overlaps = [o * BLOCK_SIZE for o in overlaps]
    total_input = sum(trace[i].input_length for i in range(1, n_pairs + 1))
 
    print(f"    Pairs analyzed:      {n}")
    print(f"    Mean block overlap:  {sum(overlaps)/n:.1f} blocks ({sum(token_overlaps)/n:.0f} tokens)")
    print(f"    Median overlap:      {overlaps_s[n//2]} blocks")
    print(f"    P90 overlap:         {overlaps_s[int(n*0.9)]} blocks")
    print(f"    Max overlap:         {overlaps_s[-1]} blocks ({overlaps_s[-1]*BLOCK_SIZE:,} tokens)")
    print(f"    Zero-overlap pairs:  {sum(1 for o in overlaps if o == 0)}/{n} ({sum(1 for o in overlaps if o == 0)/n*100:.1f}%)")
    print(f"    >=3 block overlap:   {sum(1 for o in overlaps if o >= 3)}/{n} ({sum(1 for o in overlaps if o >= 3)/n*100:.1f}%)")
    print(f"    Token overlap / input: {sum(token_overlaps)/max(1,total_input)*100:.1f}%")
 
 
# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════
 
def main():
    parser = argparse.ArgumentParser(description="Validate PoC against Mooncake production traces")
    parser.add_argument("--trace-dir", required=True,
                        help="Path to Mooncake trace directory (e.g., Mooncake/FAST25-release/traces)")
    parser.add_argument("--fast", action="store_true",
                        help="Subsample large traces for faster validation")
    parser.add_argument("--output", default="output/validation_report.json",
                        help="Output path for validation report JSON")
    args = parser.parse_args()
 
    print("=" * 70)
    print("  Adaptive KVPress PoC — Production Trace Validation")
    print("=" * 70)
 
    # ── Step 1: Find and validate trace files ────────────────────────
    print("\n▸ Step 1: Locating trace files...")
    trace_files = find_trace_files(args.trace_dir)
 
    if not trace_files:
        print(f"  ERROR: No .jsonl files found in {args.trace_dir}")
        print(f"  Expected files: trace.jsonl, conversation_trace.jsonl, etc.")
        sys.exit(1)
 
    print(f"  Found {len(trace_files)} trace file(s):")
    for name, path in trace_files.items():
        valid, msg, count = validate_trace_format(path)
        status = "OK" if valid else "FAIL"
        size_mb = path.stat().st_size / (1024 * 1024)
        print(f"    [{status}] {name:<20s} → {path.name} ({size_mb:.1f} MB, {count:,} entries)")
        if not valid:
            print(msg)
 
    # ── Step 2: Load traces ──────────────────────────────────────────
    print("\n▸ Step 2: Loading production traces...")
    real_traces: dict[str, MooncakeTrace] = {}
    for name, path in trace_files.items():
        t0 = time.time()
        trace = load_mooncake_trace(path, name=f"production_{name}")
        load_time = time.time() - t0
        real_traces[name] = trace
        summary = trace.summary()
        print(f"  {name}: {summary['num_requests']:,} requests, "
              f"mean_input={summary['mean_input_tokens']:.0f} tok, "
              f"p99={summary['p99_input']:,} tok, "
              f"blocks={summary['unique_blocks']:,} "
              f"(loaded in {load_time:.1f}s)")
 
    # ── Step 3: Compare with synthetic generator ─────────────────────
    print("\n▸ Step 3: Synthetic vs Production comparison...")
 
    variant_map = {
        "arxiv": "arxiv",
        "trace": "arxiv",  # original trace.jsonl maps to arxiv variant
        "conversation": "conversation",
        "tool_agent": "tool_agent",
    }
 
    for name, real_trace in real_traces.items():
        variant = variant_map.get(name)
        if variant:
            synthetic = generate_synthetic_mooncake_trace(
                num_requests=min(len(real_trace), 2000),
                duration_ms=int(real_trace.duration_sec * 1000),
                variant=variant,
            )
            compare_trace_stats(real_trace, synthetic, name)
        else:
            print(f"\n  [{name.upper()}] No synthetic variant configured — skipping comparison")
 
    # ── Step 4: Policy comparison on real traces ─────────────────────
    print("\n" + "=" * 70)
    print("▸ Step 4: Running policy comparison on production traces...")
    print("=" * 70)
 
    all_results = {}
    for name, trace in real_traces.items():
        print(f"\n  ── {name.upper()} trace ──")
        results = run_policy_comparison(trace, name, fast=args.fast)
        all_results[name] = results
 
    # ── Step 5: Reuse bonus sweep on primary trace ───────────────────
    print("\n" + "=" * 70)
    print("▸ Step 5: Reuse bonus impact on production trace...")
    print("=" * 70)
 
    # Use the first trace that has hash_ids
    primary_name = next(
        (n for n, t in real_traces.items() if any(e.hash_ids for e in t)),
        list(real_traces.keys())[0]
    )
    run_reuse_bonus_sweep(real_traces[primary_name], primary_name, fast=args.fast)
 
    # ── Step 6: Prefix depth analysis on all traces with hash_ids ────
    print("\n" + "=" * 70)
    print("▸ Step 6: Prefix reuse depth analysis...")
    print("=" * 70)
 
    for name, trace in real_traces.items():
        run_prefix_depth_analysis(trace, name)
 
    # ── Step 7: Generate validation report ───────────────────────────
    print("\n" + "=" * 70)
    print("▸ Step 7: Generating validation report...")
    print("=" * 70)
 
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
 
    report = {
        "validation_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "trace_dir": str(args.trace_dir),
        "traces_loaded": {
            name: trace.summary() for name, trace in real_traces.items()
        },
        "policy_results": {},
    }
 
    for name, results in all_results.items():
        report["policy_results"][name] = {
            pname: r.summary_dict() for pname, r in results.items()
        }
 
    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)
    print(f"  Saved: {args.output}")
 
    # ── Summary ──────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  VALIDATION COMPLETE")
    print("=" * 70)
    print(f"\n  Traces validated:  {len(real_traces)}")
    print(f"  Total requests:    {sum(len(t) for t in real_traces.values()):,}")
    print(f"  Policies tested:   {len(all_results.get(list(all_results.keys())[0], {}))}")
    print(f"  Report saved to:   {args.output}")
 
 
if __name__ == "__main__":
    main()
 
