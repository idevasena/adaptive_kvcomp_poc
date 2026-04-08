"""Surrogate selection policy: choose the right scorer architecture.

KVzap offers Linear (0.02% overhead, R²≈0.63) and MLP (1.1% overhead, R²≈0.77).
This module adds a tiered selection strategy:
  Tier 0: Heuristic (no surrogate, e.g. KnormPress) — for ≤1.5x compression
  Tier 1: Linear surrogate — for latency-critical long contexts
  Tier 2: MLP surrogate — for aggressive/faithful pruning
  Tier 3: MLP + CriticalKV refinement — for maximum quality
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum


class SurrogateTier(IntEnum):
    HEURISTIC = 0  # Training-free scorer (KnormPress, ExpectedAttention)
    LINEAR = 1  # KVzap-Linear (0.02% overhead)
    MLP = 2  # KVzap-MLP (1.1% overhead)
    ENSEMBLE = 3  # MLP + CriticalKV refinement


@dataclass
class SurrogateTierInfo:
    tier: SurrogateTier
    name: str
    flop_overhead_pct: float  # % of linear projection FLOPs
    memory_overhead_pct: float
    r2_typical: float  # typical R² with oracle


TIER_INFO = {
    SurrogateTier.HEURISTIC: SurrogateTierInfo(
        SurrogateTier.HEURISTIC, "heuristic", 0.0, 0.0, 0.50
    ),
    SurrogateTier.LINEAR: SurrogateTierInfo(
        SurrogateTier.LINEAR, "linear", 0.02, 0.02, 0.63
    ),
    SurrogateTier.MLP: SurrogateTierInfo(
        SurrogateTier.MLP, "mlp", 1.1, 1.1, 0.77
    ),
    SurrogateTier.ENSEMBLE: SurrogateTierInfo(
        SurrogateTier.ENSEMBLE, "ensemble", 2.5, 2.0, 0.85
    ),
}


class SurrogateSelector:
    """Selects the appropriate surrogate tier based on workload context."""

    def __init__(self, default_tier: SurrogateTier = SurrogateTier.MLP):
        self.default_tier = default_tier

    def select_tier(
        self,
        target_compression: float,
        sequence_length: int,
        task_type: str = "general",
        latency_budget_ms: float = 100.0,
    ) -> SurrogateTier:
        """Select surrogate tier based on workload parameters.

        Decision tree:
          - Very low compression (≤1.5x): heuristic is sufficient
          - Latency-critical + long context: linear to minimize overhead
          - Aggressive compression + faithful task: MLP or ensemble
          - Default: MLP (best quality/cost tradeoff)
        """
        # Very low compression — heuristic is enough
        if target_compression <= 1.5:
            return SurrogateTier.HEURISTIC

        # Extreme quality requirement
        if task_type in ("reasoning", "code", "math") and target_compression > 3.0:
            return SurrogateTier.ENSEMBLE

        # Aggressive compression needs MLP
        if target_compression > 2.5:
            return SurrogateTier.MLP

        # Latency-critical long contexts — use linear
        if sequence_length > 65536 and latency_budget_ms < 50:
            return SurrogateTier.LINEAR

        # Faithful tasks prefer MLP
        if task_type in ("reasoning", "code", "multi_hop_qa"):
            return SurrogateTier.MLP

        # Default to MLP for balanced workloads
        if target_compression > 1.5:
            return SurrogateTier.MLP

        return self.default_tier

    def get_tier_info(self, tier: SurrogateTier) -> SurrogateTierInfo:
        return TIER_INFO[tier]

    def estimate_overhead(self, tier: SurrogateTier, seq_length: int) -> dict:
        """Estimate compute and memory overhead for a given tier and length."""
        info = TIER_INFO[tier]
        # Scale overhead with sequence length (attention is quadratic)
        # But surrogate overhead is linear, so relative cost shrinks
        attn_flops = seq_length * seq_length  # simplified
        surrogate_flops = seq_length * (info.flop_overhead_pct / 100.0) * attn_flops
        return {
            "tier": info.name,
            "flop_overhead_pct": info.flop_overhead_pct,
            "memory_overhead_pct": info.memory_overhead_pct,
            "r2_expected": info.r2_typical,
            "estimated_surrogate_flops": surrogate_flops,
        }
