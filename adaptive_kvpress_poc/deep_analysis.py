#!/usr/bin/env python3
"""Deep analysis of trace-driven simulation results for evaluation document."""

import json
import sys
import os
import math

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PARENT_DIR = os.path.dirname(_THIS_DIR)
if _PARENT_DIR not in sys.path:
    sys.path.insert(0, _PARENT_DIR)

from adaptive_kvpress_poc.traces import generate_synthetic_mooncake_trace, generate_synthetic_azure_trace
from adaptive_kvpress_poc.policies import CompressionPolicy
from adaptive_kvpress_poc.simulator import TraceSimulator, SimulationConfig

# ── Experiment 1: Trace Statistical Analysis ─────────────────────────
print("=" * 70)
print("EXPERIMENT 1: Trace Statistical Characterization")
print("=" * 70)

traces = {}
for variant in ["arxiv", "conversation", "tool_agent"]:
    traces[variant] = generate_synthetic_mooncake_trace(500, 300000, variant)
traces["azure"] = generate_synthetic_azure_trace(250, 300000)

for name, t in traces.items():
    s = t.summary()
    inputs = sorted([e.input_length for e in t])
    outputs = sorted([e.output_length for e in t])
    n = len(inputs)

    # Block reuse analysis
    bf = t.block_frequency
    if bf:
        freqs = sorted(bf.values(), reverse=True)
        hot_blocks = sum(1 for f in freqs if f >= 10)
        cold_blocks = sum(1 for f in freqs if f == 1)
        top10_refs = sum(freqs[:10])
        total_refs = sum(freqs)
    else:
        hot_blocks = cold_blocks = top10_refs = total_refs = 0

    print(f"\n  [{name.upper()}]")
    print(f"    Requests:       {n}")
    print(f"    Duration:       {s['duration_sec']:.0f} sec ({s['duration_sec']/60:.1f} min)")
    print(f"    Request rate:   {n / max(1, s['duration_sec']) * 60:.1f} req/min")
    print(f"    Input tokens:   mean={s['mean_input_tokens']:.0f}  p50={inputs[n//2]}  p90={inputs[int(n*0.9)]}  p99={s['p99_input']}  max={s['max_input']}")
    print(f"    Output tokens:  mean={s['mean_output_tokens']:.0f}  p50={outputs[n//2]}  p90={outputs[int(n*0.9)]}")
    print(f"    Total tokens:   {sum(e.total_tokens for e in t):,}")
    if bf:
        print(f"    Unique blocks:  {len(bf)}")
        print(f"    Block refs:     {total_refs:,} total")
        print(f"    Hot blocks:     {hot_blocks} (freq>=10), covering {sum(f for f in freqs if f >= 10)/max(1,total_refs)*100:.1f}% of refs")
        print(f"    Cold blocks:    {cold_blocks} (freq=1), covering {cold_blocks/max(1,total_refs)*100:.1f}% of refs")
        print(f"    Top-10 blocks:  {top10_refs} refs ({top10_refs/max(1,total_refs)*100:.1f}% of all)")
        print(f"    Skewness:       top-10 blocks serve {top10_refs/max(1,total_refs)*100:.1f}% of all block references")

# ── Experiment 2: Score Distribution Analysis ────────────────────────
print("\n" + "=" * 70)
print("EXPERIMENT 2: Score Distribution Analysis per Input Length")
print("=" * 70)

from adaptive_kvpress_poc.score_profiles import ScoreProfileEngine

engine = ScoreProfileEngine("qwen3-8b", seed=42)
for length in [1024, 4096, 8192, 16384, 32768]:
    dist = engine.generate_scores(length, surrogate_tier=2)
    ss = sorted(dist.scores)
    n = len(ss)
    q = lambda p: ss[int(n * p)]
    above_zero = sum(1 for s in ss if s >= 0) / n

    print(f"\n  Input length = {length:,} tokens:")
    print(f"    Score mean={dist.global_mean:.3f}  std={dist.global_std:.3f}  entropy={dist.entropy:.3f}")
    print(f"    Quantiles:  q10={q(0.1):.3f}  q25={q(0.25):.3f}  q50={q(0.5):.3f}  q75={q(0.75):.3f}  q90={q(0.9):.3f}")
    print(f"    Fraction above 0: {above_zero:.3f}  ({above_zero*100:.1f}%)")
    print(f"    At tau=-1.0: retain {dist.fraction_above(-1.0)*100:.1f}%  -> {1/max(0.01, dist.fraction_above(-1.0)):.2f}x CR")
    print(f"    At tau=-0.5: retain {dist.fraction_above(-0.5)*100:.1f}%  -> {1/max(0.01, dist.fraction_above(-0.5)):.2f}x CR")
    print(f"    At tau= 0.0: retain {dist.fraction_above(0.0)*100:.1f}%  -> {1/max(0.01, dist.fraction_above(0.0)):.2f}x CR")

# ── Experiment 3: Per-Policy Detailed Breakdown ──────────────────────
print("\n" + "=" * 70)
print("EXPERIMENT 3: Per-Policy Detailed Analysis (ArXiv Trace)")
print("=" * 70)

policies = {
    "fixed_tau=-1.0": CompressionPolicy.fixed_baseline(tau=-1.0),
    "fixed_tau=-0.5": CompressionPolicy.fixed_baseline(tau=-0.5),
    "percentile_50%": CompressionPolicy.percentile_baseline(0.5),
    "adaptive_2x": CompressionPolicy.balanced(target_compression=2.0),
    "adaptive_3x": CompressionPolicy.balanced(target_compression=3.0),
    "conservative": CompressionPolicy.conservative(),
    "latency_optimal": CompressionPolicy.latency_optimal(),
}

arxiv = traces["arxiv"]
bf = arxiv.block_frequency
sim = TraceSimulator(SimulationConfig(store_per_request=True, report_interval=9999))

for pname, policy in policies.items():
    result = sim.run(arxiv, policy, verbose=False)
    crs = result.compression_ratios
    n = len(crs)
    crs_s = sorted(crs)
    thresholds = [r.threshold_used for r in result.per_request]
    reuses = [r.reuse_efficiency for r in result.per_request]
    mem_saved = [1 - r.memory_bytes_compressed / max(1, r.memory_bytes_full) for r in result.per_request]

    # Correlate CR with input length
    lengths = [r.input_length for r in result.per_request]
    short = [r.compression_ratio for r in result.per_request if r.input_length < 4000]
    medium = [r.compression_ratio for r in result.per_request if 4000 <= r.input_length < 16000]
    long = [r.compression_ratio for r in result.per_request if r.input_length >= 16000]

    print(f"\n  [{pname}]")
    print(f"    CR distribution:  mean={result.mean_compression:.2f}x  std={result.std_compression:.2f}  CV={result.cv_compression:.3f}")
    print(f"    CR percentiles:   p5={crs_s[int(n*0.05)]:.2f}x  p25={crs_s[n//4]:.2f}x  p50={crs_s[n//2]:.2f}x  p75={crs_s[int(n*0.75)]:.2f}x  p95={crs_s[int(n*0.95)]:.2f}x")
    print(f"    Memory:           full={result.total_memory_full_gb:.1f} GB  compressed={result.total_memory_compressed_gb:.1f} GB  saved={result.memory_savings_pct:.1f}%")
    print(f"    Prefix reuse:     {result.total_prefix_preserved}/{result.total_prefix_blocks} blocks preserved ({result.mean_reuse_efficiency*100:.1f}%)")
    print(f"    Threshold stats:  mean={sum(thresholds)/len(thresholds):.3f}  std={((sum((t-sum(thresholds)/len(thresholds))**2 for t in thresholds)/len(thresholds))**0.5):.3f}")
    print(f"    CR by length:     short(<4k)={sum(short)/max(1,len(short)):.2f}x [{len(short)} req]  medium(4-16k)={sum(medium)/max(1,len(medium)):.2f}x [{len(medium)} req]  long(>16k)={sum(long)/max(1,len(long)):.2f}x [{len(long)} req]")

# ── Experiment 4: Reuse Bonus Impact Isolation ───────────────────────
print("\n" + "=" * 70)
print("EXPERIMENT 4: Reuse-Aware Bonus Impact Isolation")
print("=" * 70)

from adaptive_kvpress_poc.policies.threshold import AdaptiveThresholdPolicy
from adaptive_kvpress_poc.policies.surrogate import SurrogateSelector, SurrogateTier
from adaptive_kvpress_poc.policies.window import AdaptiveWindowPolicy, WindowConfig
from adaptive_kvpress_poc.score_profiles import ScoreProfileEngine

for bonus_scale in [0.0, 0.10, 0.25, 0.50, 0.75]:
    p = CompressionPolicy(
        threshold_policy=AdaptiveThresholdPolicy(target_compression=2.0),
        surrogate_selector=SurrogateSelector(SurrogateTier.MLP),
        window_policy=AdaptiveWindowPolicy(WindowConfig(base_window=128, head_tokens=4,
            scale_with_compression=True, score_aware_boundary=True)),
        score_engine=ScoreProfileEngine("qwen3-8b"),
        reuse_aware=(bonus_scale > 0),
        reuse_bonus_scale=bonus_scale,
    )
    r = sim.run(arxiv, p, verbose=False)
    print(f"  bonus_scale={bonus_scale:.2f}  →  CR={r.mean_compression:.2f}x  CV={r.cv_compression:.3f}  reuse={r.mean_reuse_efficiency:.3f} ({r.mean_reuse_efficiency*100:.1f}%)  mem_saved={r.memory_savings_pct:.1f}%")

# ── Experiment 5: Threshold Sensitivity ──────────────────────────────
print("\n" + "=" * 70)
print("EXPERIMENT 5: Fixed Threshold Sweep (-2.0 to +1.5)")
print("=" * 70)

for tau in [-2.0, -1.5, -1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0, 1.5]:
    p = CompressionPolicy.fixed_baseline(tau=tau)
    r = sim.run(arxiv, p, verbose=False)
    print(f"  tau={tau:+5.2f}  →  CR={r.mean_compression:.2f}x  std={r.std_compression:.2f}  CV={r.cv_compression:.3f}  mem={r.memory_savings_pct:.1f}%  reuse={r.mean_reuse_efficiency*100:.1f}%")

# ── Experiment 6: Memory Footprint by Model Architecture ─────────────
print("\n" + "=" * 70)
print("EXPERIMENT 6: Memory Footprint by Model Architecture")
print("=" * 70)

from adaptive_kvpress_poc.score_profiles import MODEL_CONFIGS

for model_name, cfg in MODEL_CONFIGS.items():
    kv_per_token = 2 * cfg.num_layers * cfg.num_kv_heads * cfg.head_dim * 2  # bf16
    for seq_len in [4096, 16384, 65536, 131072]:
        full_mb = seq_len * kv_per_token / (1024**2)
        at_2x = full_mb / 2
        at_4x = full_mb / 4
        print(f"  {model_name:15s}  seq={seq_len:>7,}  full={full_mb:>8.1f} MB  at_2x={at_2x:>8.1f} MB  at_4x={at_4x:>8.1f} MB")

# ── Experiment 7: Prefix Reuse Depth Analysis ────────────────────────
print("\n" + "=" * 70)
print("EXPERIMENT 7: Prefix Reuse Depth Analysis (ArXiv Trace)")
print("=" * 70)

# Compute pairwise prefix overlap for sequential requests
overlaps = []
for i in range(1, min(200, len(arxiv))):
    overlap = arxiv[i].prefix_overlap(arxiv[i-1])
    overlaps.append(overlap)

if overlaps:
    overlaps_s = sorted(overlaps)
    n = len(overlaps)
    print(f"  Sequential prefix overlap (first 200 request pairs):")
    print(f"    mean={sum(overlaps)/n:.1f} blocks  p50={overlaps_s[n//2]} blocks  p90={overlaps_s[int(n*0.9)]} blocks  max={overlaps_s[-1]} blocks")
    print(f"    zero-overlap pairs: {sum(1 for o in overlaps if o == 0)}/{n} ({sum(1 for o in overlaps if o == 0)/n*100:.1f}%)")
    print(f"    >=3 block overlap:  {sum(1 for o in overlaps if o >= 3)}/{n} ({sum(1 for o in overlaps if o >= 3)/n*100:.1f}%)")

    # Translate to tokens
    BLOCK_SIZE = 512
    token_overlaps = [o * BLOCK_SIZE for o in overlaps]
    print(f"    mean token overlap: {sum(token_overlaps)/len(token_overlaps):.0f} tokens")
    print(f"    This represents {sum(token_overlaps)/sum(arxiv[i].input_length for i in range(1, min(200, len(arxiv))))*100:.1f}% of input tokens on average")

print("\n" + "=" * 70)
print("ALL EXPERIMENTS COMPLETE")
print("=" * 70)
