"""Integrated compression policy combining all three adaptive axes.

This is the main policy engine that:
  1. Selects surrogate tier (§4)
  2. Generates / receives score distribution
  3. Selects threshold τ (§3)
  4. Computes window keep mask (§5)
  5. Applies reuse-aware score bonus (§7.6)
  6. Returns compression decisions
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from ..score_profiles import ScoreDistribution, ScoreProfileEngine, ModelConfig
from ..traces.loader import TraceEntry
from .threshold import (
    ThresholdPolicy,
    ScoreStats,
    FixedThresholdPolicy,
    AdaptiveThresholdPolicy,
    PercentileThresholdPolicy,
)
from .surrogate import SurrogateSelector, SurrogateTier, TIER_INFO
from .window import AdaptiveWindowPolicy, WindowConfig

BLOCK_SIZE = 512


@dataclass
class CompressionResult:
    """Result of applying compression policy to a single request."""

    # Input info
    input_length: int
    timestamp: float

    # Compression decisions
    tokens_retained: int
    tokens_evicted: int
    compression_ratio: float

    # Policy decisions
    threshold_used: float
    surrogate_tier: int
    window_size: int

    # Prefix reuse analysis
    prefix_blocks_total: int = 0
    prefix_blocks_preserved: int = 0  # blocks fully retained after compression
    reuse_efficiency: float = 1.0  # fraction of prefix blocks still usable

    # Score statistics
    score_mean: float = 0.0
    score_std: float = 0.0
    score_entropy: float = 0.0

    # Memory estimate (bytes, assuming bf16 KV for all layers/heads)
    memory_bytes_full: int = 0
    memory_bytes_compressed: int = 0


@dataclass
class CompressionPolicy:
    """Full adaptive compression policy combining all three axes.

    Usage:
        policy = CompressionPolicy.balanced(target_compression=2.0)
        result = policy.compress(trace_entry, block_frequency)
    """

    threshold_policy: ThresholdPolicy
    surrogate_selector: SurrogateSelector
    window_policy: AdaptiveWindowPolicy
    score_engine: ScoreProfileEngine
    reuse_aware: bool = True  # enable reuse-aware score bonus
    reuse_bonus_scale: float = 0.25

    # --- named constructors ------------------------------------------------

    @classmethod
    def fixed_baseline(
        cls,
        tau: float = -1.0,
        model: str = "qwen3-8b",
    ) -> "CompressionPolicy":
        """KVzap baseline: fixed threshold, MLP surrogate, 128-token window."""
        return cls(
            threshold_policy=FixedThresholdPolicy(tau=tau),
            surrogate_selector=SurrogateSelector(SurrogateTier.MLP),
            window_policy=AdaptiveWindowPolicy(
                WindowConfig(base_window=128, head_tokens=0,
                             scale_with_compression=False,
                             score_aware_boundary=False)
            ),
            score_engine=ScoreProfileEngine(model),
            reuse_aware=False,
        )

    @classmethod
    def conservative(
        cls,
        model: str = "qwen3-8b",
    ) -> "CompressionPolicy":
        """Conservative: low compression, large window, MLP."""
        return cls(
            threshold_policy=AdaptiveThresholdPolicy(target_compression=1.5),
            surrogate_selector=SurrogateSelector(SurrogateTier.MLP),
            window_policy=AdaptiveWindowPolicy(
                WindowConfig(base_window=256, head_tokens=4,
                             scale_with_compression=True,
                             score_aware_boundary=True,
                             task_type="general")
            ),
            score_engine=ScoreProfileEngine(model),
            reuse_aware=True,
        )

    @classmethod
    def balanced(
        cls,
        target_compression: float = 2.0,
        model: str = "qwen3-8b",
        task_type: str = "general",
    ) -> "CompressionPolicy":
        """Balanced: adaptive threshold, MLP, scaled window."""
        return cls(
            threshold_policy=AdaptiveThresholdPolicy(
                target_compression=target_compression
            ),
            surrogate_selector=SurrogateSelector(SurrogateTier.MLP),
            window_policy=AdaptiveWindowPolicy(
                WindowConfig(base_window=128, head_tokens=4,
                             scale_with_compression=True,
                             score_aware_boundary=True,
                             task_type=task_type)
            ),
            score_engine=ScoreProfileEngine(model),
            reuse_aware=True,
        )

    @classmethod
    def aggressive(
        cls,
        model: str = "qwen3-8b",
    ) -> "CompressionPolicy":
        """Aggressive: high compression target, ensemble scorer."""
        return cls(
            threshold_policy=AdaptiveThresholdPolicy(target_compression=3.5),
            surrogate_selector=SurrogateSelector(SurrogateTier.ENSEMBLE),
            window_policy=AdaptiveWindowPolicy(
                WindowConfig(base_window=192, head_tokens=4,
                             scale_with_compression=True,
                             score_aware_boundary=True,
                             task_type="general")
            ),
            score_engine=ScoreProfileEngine(model),
            reuse_aware=True,
        )

    @classmethod
    def latency_optimal(
        cls,
        model: str = "qwen3-8b",
    ) -> "CompressionPolicy":
        """Latency-optimal: linear surrogate, minimal window."""
        return cls(
            threshold_policy=AdaptiveThresholdPolicy(target_compression=2.0),
            surrogate_selector=SurrogateSelector(SurrogateTier.LINEAR),
            window_policy=AdaptiveWindowPolicy(
                WindowConfig(base_window=128, head_tokens=4,
                             scale_with_compression=False,
                             score_aware_boundary=False)
            ),
            score_engine=ScoreProfileEngine(model),
            reuse_aware=False,
        )

    @classmethod
    def percentile_baseline(
        cls,
        prune_fraction: float = 0.5,
        model: str = "qwen3-8b",
    ) -> "CompressionPolicy":
        """Percentile-based: prune bottom p% of scores."""
        return cls(
            threshold_policy=PercentileThresholdPolicy(prune_fraction=prune_fraction),
            surrogate_selector=SurrogateSelector(SurrogateTier.MLP),
            window_policy=AdaptiveWindowPolicy(
                WindowConfig(base_window=128, head_tokens=0,
                             scale_with_compression=False,
                             score_aware_boundary=False)
            ),
            score_engine=ScoreProfileEngine(model),
            reuse_aware=False,
        )

    # --- compression logic -------------------------------------------------

    def compress(
        self,
        entry: TraceEntry,
        block_frequency: dict[int, int] | None = None,
    ) -> CompressionResult:
        """Apply compression policy to a single trace entry.

        Args:
            entry: Trace entry with input_length, hash_ids, etc.
            block_frequency: Global block hash → usage count map
                             (from MooncakeTrace.block_frequency)
        """
        n = entry.input_length

        # 1. Select surrogate tier
        tier = self.surrogate_selector.select_tier(
            target_compression=getattr(
                self.threshold_policy, "target_compression", 2.0
            ),
            sequence_length=n,
        )

        # 2. Generate score distribution
        score_dist = self.score_engine.generate_scores(n, surrogate_tier=tier)
        scores = list(score_dist.scores)  # mutable copy

        # 3. Apply reuse-aware score bonus
        if self.reuse_aware and block_frequency and entry.hash_ids:
            scores = self._apply_reuse_bonus(scores, entry, block_frequency)

        # 4. Compute score statistics
        sorted_scores = sorted(scores)
        ns = len(sorted_scores)
        stats = ScoreStats(
            mean=score_dist.global_mean,
            std=score_dist.global_std,
            entropy=score_dist.entropy,
            q10=sorted_scores[int(ns * 0.10)] if ns else 0,
            q25=sorted_scores[int(ns * 0.25)] if ns else 0,
            q50=sorted_scores[ns // 2] if ns else 0,
            q75=sorted_scores[int(ns * 0.75)] if ns else 0,
            q90=sorted_scores[int(ns * 0.90)] if ns else 0,
            num_tokens=ns,
        )

        # 5. Select threshold
        threshold = self.threshold_policy.select_threshold(stats)

        # 6. Compute window keep mask (need approximate compression for scaling)
        approx_cr = 1.0 / max(0.01, score_dist.fraction_above(threshold))
        keep_mask = self.window_policy.compute_keep_mask(n, approx_cr, scores)

        # 7. Apply threshold + window → retention decisions
        tokens_retained = 0
        for i in range(n):
            if keep_mask[i] or (i < len(scores) and scores[i] >= threshold):
                tokens_retained += 1

        tokens_evicted = n - tokens_retained
        compression_ratio = n / max(1, tokens_retained)

        # 8. Prefix reuse analysis
        prefix_total, prefix_preserved = self._analyze_prefix_reuse(
            entry, scores, threshold, keep_mask
        )

        # 9. Memory estimate (bf16, all layers/heads)
        config = self.score_engine.config
        kv_per_token = 2 * config.num_layers * config.num_kv_heads * config.head_dim * 2  # bf16
        memory_full = n * kv_per_token
        memory_compressed = tokens_retained * kv_per_token

        return CompressionResult(
            input_length=n,
            timestamp=entry.timestamp,
            tokens_retained=tokens_retained,
            tokens_evicted=tokens_evicted,
            compression_ratio=compression_ratio,
            threshold_used=threshold,
            surrogate_tier=tier,
            window_size=self.window_policy.get_effective_window_size(compression_ratio),
            prefix_blocks_total=prefix_total,
            prefix_blocks_preserved=prefix_preserved,
            reuse_efficiency=prefix_preserved / max(1, prefix_total),
            score_mean=stats.mean,
            score_std=stats.std,
            score_entropy=stats.entropy,
            memory_bytes_full=memory_full,
            memory_bytes_compressed=memory_compressed,
        )

    def _apply_reuse_bonus(
        self,
        scores: list[float],
        entry: TraceEntry,
        block_frequency: dict[int, int],
    ) -> list[float]:
        """Boost scores of tokens in frequently-reused prefix blocks."""
        for block_idx, hash_id in enumerate(entry.hash_ids):
            reuse_count = block_frequency.get(hash_id, 1)
            if reuse_count > 1:
                bonus = math.log1p(reuse_count) * self.reuse_bonus_scale
                start = block_idx * BLOCK_SIZE
                end = min(start + BLOCK_SIZE, len(scores))
                for i in range(start, end):
                    scores[i] += bonus
        return scores

    def _analyze_prefix_reuse(
        self,
        entry: TraceEntry,
        scores: list[float],
        threshold: float,
        keep_mask: list[bool],
    ) -> tuple[int, int]:
        """Analyze how many prefix blocks survive compression intact."""
        if not entry.hash_ids:
            return 0, 0

        total_blocks = len(entry.hash_ids)
        preserved = 0

        for block_idx in range(total_blocks):
            start = block_idx * BLOCK_SIZE
            end = min(start + BLOCK_SIZE, len(scores))
            if end <= start:
                break

            # A block is "preserved" if >80% of its tokens are retained
            retained_in_block = 0
            block_size = end - start
            for i in range(start, end):
                if keep_mask[i] or (i < len(scores) and scores[i] >= threshold):
                    retained_in_block += 1

            if retained_in_block >= 0.8 * block_size:
                preserved += 1

        return total_blocks, preserved

    @property
    def name(self) -> str:
        parts = [
            self.threshold_policy.name,
            f"surrogate={self.surrogate_selector.default_tier.name.lower()}",
            self.window_policy.name,
        ]
        if self.reuse_aware:
            parts.append("reuse-aware")
        return " | ".join(parts)
