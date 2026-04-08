"""Threshold calibration policies for KVzap-style score-based pruning.

KVzap uses a fixed threshold τ to prune KV pairs with scores below τ.
The same τ yields ~20% compression ratio variation across prompts.
These policies aim to stabilize compression at a target quality level.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class ScoreStats:
    """Lightweight statistics from a score distribution."""

    mean: float
    std: float
    entropy: float
    q10: float  # 10th percentile
    q25: float
    q50: float  # median
    q75: float
    q90: float
    num_tokens: int


class ThresholdPolicy(ABC):
    """Base class for threshold selection policies."""

    @abstractmethod
    def select_threshold(self, stats: ScoreStats) -> float:
        """Select pruning threshold based on score statistics."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable policy name."""


class FixedThresholdPolicy(ThresholdPolicy):
    """Fixed threshold τ — the KVzap baseline approach.

    Higher τ → more aggressive pruning → higher compression.
    """

    def __init__(self, tau: float = -1.0):
        self.tau = tau

    def select_threshold(self, stats: ScoreStats) -> float:
        return self.tau

    @property
    def name(self) -> str:
        return f"fixed(τ={self.tau:.2f})"


class AdaptiveThresholdPolicy(ThresholdPolicy):
    """Score-distribution-aware threshold selection.

    Uses the score statistics to place τ relative to the distribution,
    achieving more stable compression ratios across diverse prompts.

    The threshold is computed as:
        τ = mean + k * std

    where k is calibrated to achieve the target compression ratio on
    a reference workload. The entropy term adjusts for information
    density: high entropy → diverse importance → more conservative.
    """

    def __init__(
        self,
        target_compression: float = 2.0,
        entropy_sensitivity: float = 0.15,
        base_k: float = 0.0,
    ):
        self.target_compression = target_compression
        self.entropy_sensitivity = entropy_sensitivity
        self.base_k = base_k

        # Calibration: map target compression → k offset
        # Tuned against Mooncake arxiv trace score distributions
        self._compression_to_k = {
            1.5: -0.3,
            2.0: 0.1,
            2.5: 0.4,
            3.0: 0.7,
            3.5: 0.9,
            4.0: 1.2,
        }

    def select_threshold(self, stats: ScoreStats) -> float:
        # Interpolate k from target compression
        k = self.base_k
        for target, k_val in sorted(self._compression_to_k.items()):
            if self.target_compression <= target:
                k = k_val
                break
        else:
            k = 0.5  # very aggressive

        # Entropy adjustment: high entropy → reduce k (keep more tokens)
        # Normalized entropy for typical distributions is ~2-4
        entropy_factor = max(0, stats.entropy - 2.5) * self.entropy_sensitivity
        k -= entropy_factor

        threshold = stats.mean + k * stats.std
        return threshold

    @property
    def name(self) -> str:
        return f"adaptive(target={self.target_compression:.1f}x)"


class PercentileThresholdPolicy(ThresholdPolicy):
    """Percentile-based threshold — prune bottom p% of scores.

    More stable compression than fixed τ because it adapts to the
    score distribution shape. Equivalent to top-k but expressed as
    a fraction.
    """

    def __init__(self, prune_fraction: float = 0.5):
        """prune_fraction: fraction of tokens to prune (0.5 → 2x compression)."""
        self.prune_fraction = max(0.0, min(0.95, prune_fraction))

    def select_threshold(self, stats: ScoreStats) -> float:
        # Map prune fraction to approximate quantile threshold
        # Using the stats quantiles as anchors
        if self.prune_fraction <= 0.10:
            return stats.q10 - stats.std
        elif self.prune_fraction <= 0.25:
            return stats.q10 + (stats.q25 - stats.q10) * (
                (self.prune_fraction - 0.10) / 0.15
            )
        elif self.prune_fraction <= 0.50:
            return stats.q25 + (stats.q50 - stats.q25) * (
                (self.prune_fraction - 0.25) / 0.25
            )
        elif self.prune_fraction <= 0.75:
            return stats.q50 + (stats.q75 - stats.q50) * (
                (self.prune_fraction - 0.50) / 0.25
            )
        else:
            return stats.q75 + (stats.q90 - stats.q75) * (
                (self.prune_fraction - 0.75) / 0.25
            )

    @property
    def name(self) -> str:
        return f"percentile(prune={self.prune_fraction:.0%})"
