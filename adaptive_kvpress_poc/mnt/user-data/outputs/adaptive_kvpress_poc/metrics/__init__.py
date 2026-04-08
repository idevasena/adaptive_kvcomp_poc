"""Metrics collection and report generation.

Produces structured JSON reports from simulation results,
suitable for dashboard visualization and cross-experiment comparison.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from ..simulator import SimulationResult


def generate_comparison_report(
    results: dict[str, SimulationResult],
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Generate a structured comparison report across policies.

    Returns a JSON-serializable dict and optionally writes to disk.
    """
    report: dict[str, Any] = {
        "meta": {
            "trace": next(iter(results.values())).trace_name if results else "unknown",
            "num_policies": len(results),
            "num_requests": next(iter(results.values())).num_requests if results else 0,
        },
        "summary_table": [],
        "compression_distributions": {},
        "reuse_analysis": {},
        "memory_impact": {},
        "timelines": {},
    }

    for name, sim in results.items():
        # Summary row
        report["summary_table"].append({
            "policy": name,
            **sim.summary_dict(),
        })

        # Compression distribution (histogram bins)
        crs = sim.compression_ratios
        if crs:
            hist = _compute_histogram(crs, bins=20, min_val=1.0, max_val=6.0)
            report["compression_distributions"][name] = hist

        # Reuse analysis
        report["reuse_analysis"][name] = {
            "mean_reuse_efficiency": round(sim.mean_reuse_efficiency, 4),
            "total_prefix_blocks": sim.total_prefix_blocks,
            "total_prefix_preserved": sim.total_prefix_preserved,
            "preservation_rate": round(
                sim.total_prefix_preserved / max(1, sim.total_prefix_blocks), 4
            ),
        }

        # Memory impact
        report["memory_impact"][name] = {
            "total_full_gb": round(sim.total_memory_full_gb, 3),
            "total_compressed_gb": round(sim.total_memory_compressed_gb, 3),
            "savings_pct": round(sim.memory_savings_pct, 1),
            "savings_gb": round(
                sim.total_memory_full_gb - sim.total_memory_compressed_gb, 3
            ),
        }

        # Timeline (sampled)
        report["timelines"][name] = {
            "timestamps_ms": sim.memory_timeline_ms[:200],
            "memory_gb": [round(m, 6) for m in sim.memory_timeline_gb[:200]],
            "compression_ratio": [round(c, 3) for c in sim.compression_timeline[:200]],
        }

    if output_path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(report, f, indent=2)

    return report


def _compute_histogram(
    values: list[float],
    bins: int = 20,
    min_val: float | None = None,
    max_val: float | None = None,
) -> dict:
    """Compute histogram of values for visualization."""
    if not values:
        return {"bin_edges": [], "counts": [], "density": []}

    lo = min_val if min_val is not None else min(values)
    hi = max_val if max_val is not None else max(values)
    if hi <= lo:
        hi = lo + 1.0

    bin_width = (hi - lo) / bins
    counts = [0] * bins
    for v in values:
        b = int((v - lo) / bin_width)
        b = max(0, min(b, bins - 1))
        counts[b] += 1

    total = len(values)
    edges = [round(lo + i * bin_width, 3) for i in range(bins + 1)]
    density = [round(c / total, 4) for c in counts]

    return {"bin_edges": edges, "counts": counts, "density": density}


def print_comparison_table(results: dict[str, SimulationResult]) -> str:
    """Print a formatted comparison table."""
    headers = [
        "Policy", "Mean CR", "Std CR", "CV", "Mem Save%",
        "Reuse Eff", "Threshold", "Window",
    ]
    rows = []
    for name, sim in results.items():
        rows.append([
            name[:30],
            f"{sim.mean_compression:.2f}x",
            f"{sim.std_compression:.2f}",
            f"{sim.cv_compression:.3f}",
            f"{sim.memory_savings_pct:.1f}%",
            f"{sim.mean_reuse_efficiency:.3f}",
            f"{sim.mean_threshold:.2f}",
            f"{sim.mean_window_size:.0f}",
        ])

    # Format as aligned table
    col_widths = [max(len(h), max(len(r[i]) for r in rows)) for i, h in enumerate(headers)]
    sep = "+" + "+".join("-" * (w + 2) for w in col_widths) + "+"
    header_line = "|" + "|".join(f" {h:^{w}} " for h, w in zip(headers, col_widths)) + "|"

    lines = [sep, header_line, sep]
    for row in rows:
        line = "|" + "|".join(f" {c:^{w}} " for c, w in zip(row, col_widths)) + "|"
        lines.append(line)
    lines.append(sep)

    table = "\n".join(lines)
    print(table)
    return table
