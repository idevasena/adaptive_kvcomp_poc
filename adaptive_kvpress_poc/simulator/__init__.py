"""Trace replay simulator: runs compression policies over trace data.

This is the core simulation engine. For each trace entry, it:
  1. Applies the compression policy (threshold + window + reuse bonus)
  2. Collects per-request metrics
  3. Tracks system-level state (memory timeline, prefix cache)
  4. Produces aggregate results for analysis
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from ..traces.loader import MooncakeTrace, TraceEntry
from ..policies.compression import CompressionPolicy, CompressionResult


@dataclass
class SimulationConfig:
    """Configuration for a trace replay simulation run."""

    name: str = "default"
    # Memory budget (bytes) — 0 means unlimited
    memory_budget_bytes: int = 0
    # Whether to track per-request results (memory intensive for large traces)
    store_per_request: bool = True
    # Progress reporting interval
    report_interval: int = 500


@dataclass
class SimulationResult:
    """Aggregate results from a complete trace replay."""

    config_name: str
    policy_name: str
    trace_name: str
    num_requests: int
    wall_time_sec: float

    # Compression statistics
    compression_ratios: list[float] = field(default_factory=list)
    mean_compression: float = 0.0
    std_compression: float = 0.0
    min_compression: float = 0.0
    max_compression: float = 0.0
    cv_compression: float = 0.0  # coefficient of variation

    # Memory statistics
    memory_savings_pct: float = 0.0  # average memory saved
    total_memory_full_gb: float = 0.0
    total_memory_compressed_gb: float = 0.0

    # Prefix reuse statistics
    mean_reuse_efficiency: float = 0.0
    total_prefix_blocks: int = 0
    total_prefix_preserved: int = 0

    # Policy breakdown
    surrogate_tier_distribution: dict[int, int] = field(default_factory=dict)
    mean_threshold: float = 0.0
    mean_window_size: float = 0.0

    # Score statistics
    mean_score_entropy: float = 0.0

    # Per-request results (optional)
    per_request: list[CompressionResult] = field(default_factory=list)

    # Timeline data (for plotting)
    memory_timeline_ms: list[float] = field(default_factory=list)
    memory_timeline_gb: list[float] = field(default_factory=list)
    compression_timeline: list[float] = field(default_factory=list)

    def summary_dict(self) -> dict:
        """Return a compact summary for reporting."""
        return {
            "policy": self.policy_name,
            "trace": self.trace_name,
            "requests": self.num_requests,
            "mean_compression": round(self.mean_compression, 2),
            "std_compression": round(self.std_compression, 2),
            "cv_compression": round(self.cv_compression, 3),
            "memory_savings_pct": round(self.memory_savings_pct, 1),
            "mean_reuse_efficiency": round(self.mean_reuse_efficiency, 3),
            "mean_threshold": round(self.mean_threshold, 3),
            "mean_window_size": round(self.mean_window_size, 1),
            "wall_time_sec": round(self.wall_time_sec, 2),
        }


class TraceSimulator:
    """Replays a Mooncake/Dynamo trace through a compression policy."""

    def __init__(self, config: SimulationConfig | None = None):
        self.config = config or SimulationConfig()

    def run(
        self,
        trace: MooncakeTrace,
        policy: CompressionPolicy,
        verbose: bool = True,
    ) -> SimulationResult:
        """Run the full trace replay simulation."""
        t0 = time.time()
        block_freq = trace.block_frequency

        results: list[CompressionResult] = []
        tier_counts: dict[int, int] = {}

        for idx, entry in enumerate(trace):
            result = policy.compress(entry, block_frequency=block_freq)
            results.append(result)

            tier_counts[result.surrogate_tier] = (
                tier_counts.get(result.surrogate_tier, 0) + 1
            )

            if verbose and (idx + 1) % self.config.report_interval == 0:
                avg_cr = sum(r.compression_ratio for r in results) / len(results)
                print(
                    f"  [{idx+1}/{len(trace)}] "
                    f"avg_compression={avg_cr:.2f}x "
                    f"last_cr={result.compression_ratio:.2f}x "
                    f"tokens={entry.input_length}"
                )

        wall_time = time.time() - t0

        # Aggregate statistics
        crs = [r.compression_ratio for r in results]
        mean_cr = sum(crs) / len(crs) if crs else 0
        std_cr = (sum((c - mean_cr) ** 2 for c in crs) / max(1, len(crs))) ** 0.5

        total_full = sum(r.memory_bytes_full for r in results)
        total_compressed = sum(r.memory_bytes_compressed for r in results)

        total_prefix = sum(r.prefix_blocks_total for r in results)
        total_preserved = sum(r.prefix_blocks_preserved for r in results)

        sim_result = SimulationResult(
            config_name=self.config.name,
            policy_name=policy.name,
            trace_name=trace.name,
            num_requests=len(results),
            wall_time_sec=wall_time,
            compression_ratios=crs,
            mean_compression=mean_cr,
            std_compression=std_cr,
            min_compression=min(crs) if crs else 0,
            max_compression=max(crs) if crs else 0,
            cv_compression=std_cr / mean_cr if mean_cr > 0 else 0,
            memory_savings_pct=(1 - total_compressed / max(1, total_full)) * 100,
            total_memory_full_gb=total_full / (1024 ** 3),
            total_memory_compressed_gb=total_compressed / (1024 ** 3),
            mean_reuse_efficiency=(
                total_preserved / max(1, total_prefix)
            ),
            total_prefix_blocks=total_prefix,
            total_prefix_preserved=total_preserved,
            surrogate_tier_distribution=tier_counts,
            mean_threshold=sum(r.threshold_used for r in results) / max(1, len(results)),
            mean_window_size=sum(r.window_size for r in results) / max(1, len(results)),
            mean_score_entropy=sum(r.score_entropy for r in results) / max(1, len(results)),
            per_request=results if self.config.store_per_request else [],
            # Timeline (sampled every 10 requests for plotting)
            memory_timeline_ms=[r.timestamp for r in results[::10]],
            memory_timeline_gb=[
                r.memory_bytes_compressed / (1024 ** 3) for r in results[::10]
            ],
            compression_timeline=[r.compression_ratio for r in results[::10]],
        )

        if verbose:
            print(f"\n  Simulation complete in {wall_time:.1f}s")
            for k, v in sim_result.summary_dict().items():
                print(f"    {k}: {v}")

        return sim_result


def run_comparison(
    trace: MooncakeTrace,
    policies: dict[str, CompressionPolicy],
    config: SimulationConfig | None = None,
    verbose: bool = True,
) -> dict[str, SimulationResult]:
    """Run multiple policies on the same trace for comparison."""
    simulator = TraceSimulator(config or SimulationConfig())
    results = {}

    for name, policy in policies.items():
        if verbose:
            print(f"\n{'='*60}")
            print(f"Running policy: {name}")
            print(f"{'='*60}")
        results[name] = simulator.run(trace, policy, verbose=verbose)

    return results
