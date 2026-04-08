"""Score Profile Engine: synthetic KVzap-like score distributions.

In a real deployment, this module would load pre-computed score profiles
from a one-time KVzap scoring run (~500 Nemotron prompts). For the PoC,
we generate synthetic score distributions that match the statistical
properties reported in the KVzap paper (arXiv:2601.07891):

  - Per-layer, per-head importance scores
  - Score distributions vary by input length and layer depth
  - R² between surrogate and oracle: 0.63 (Linear) to 0.77 (MLP)
  - Scores follow a roughly log-normal distribution with heavy tails
  - Initial tokens (attention sinks) and recent tokens have elevated scores
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelConfig:
    """Model architecture parameters relevant to KV cache scoring."""

    name: str
    num_layers: int
    num_kv_heads: int
    hidden_size: int
    head_dim: int

    # KVzap surrogate quality (R² with oracle) per tier
    r2_linear: float = 0.63
    r2_mlp: float = 0.77


# Pre-defined model configs matching KVzap paper
MODEL_CONFIGS = {
    "qwen3-8b": ModelConfig("qwen3-8b", num_layers=32, num_kv_heads=8,
                            hidden_size=4096, head_dim=128,
                            r2_linear=0.65, r2_mlp=0.77),
    "llama-3.1-8b": ModelConfig("llama-3.1-8b", num_layers=32, num_kv_heads=8,
                                hidden_size=4096, head_dim=128,
                                r2_linear=0.63, r2_mlp=0.74),
    "qwen3-32b": ModelConfig("qwen3-32b", num_layers=64, num_kv_heads=8,
                             hidden_size=5120, head_dim=128,
                             r2_linear=0.64, r2_mlp=0.76),
}


@dataclass
class ScoreDistribution:
    """Score distribution for a single request's KV cache.

    Stores per-token scores aggregated across layers and heads,
    plus per-layer statistics for detailed analysis.
    """

    # Aggregated scores: one per token position
    scores: list[float]
    seq_length: int

    # Per-layer statistics (mean, std) for surrogate tier analysis
    per_layer_mean: list[float] = field(default_factory=list)
    per_layer_std: list[float] = field(default_factory=list)

    # Aggregate stats
    global_mean: float = 0.0
    global_std: float = 0.0
    entropy: float = 0.0

    @property
    def num_tokens(self) -> int:
        return len(self.scores)

    def fraction_above(self, threshold: float) -> float:
        """Fraction of tokens with score above threshold."""
        if not self.scores:
            return 0.0
        return sum(1 for s in self.scores if s >= threshold) / len(self.scores)

    def quantile(self, q: float) -> float:
        """Return the q-th quantile (0..1) of scores."""
        if not self.scores:
            return 0.0
        sorted_scores = sorted(self.scores)
        idx = min(int(q * len(sorted_scores)), len(sorted_scores) - 1)
        return sorted_scores[idx]


class ScoreProfileEngine:
    """Generates synthetic KVzap score distributions.

    In production, this would load pre-computed profiles from a one-time
    KVzap scoring run. Here we simulate the statistical properties.
    """

    def __init__(
        self,
        model_config: ModelConfig | str = "qwen3-8b",
        seed: int = 42,
    ):
        if isinstance(model_config, str):
            model_config = MODEL_CONFIGS[model_config]
        self.config = model_config
        self.rng = random.Random(seed)

    def generate_scores(
        self,
        input_length: int,
        surrogate_tier: int = 2,  # 0=heuristic, 1=linear, 2=mlp
    ) -> ScoreDistribution:
        """Generate a synthetic score distribution for a given input length.

        Uses block-level generation with interpolation for speed.
        Models attention sink, recency bias, and surrogate noise effects.
        """
        n = input_length
        if n == 0:
            return ScoreDistribution(scores=[], seq_length=0)

        # Block-level generation for speed (one score per 32 tokens, then expand)
        block = 32
        num_blocks = max(1, (n + block - 1) // block)
        block_scores: list[float] = []

        base_mu = -0.5 - 0.3 * math.log(max(n, 1) / 1000)
        base_sigma = 0.8 + 0.1 * math.log(max(n, 1) / 1000)

        for bi in range(num_blocks):
            pos = bi * block  # representative position
            base = self.rng.gauss(base_mu, base_sigma)

            # Attention sink boost (first few blocks)
            if pos < 4:
                base += 3.0
            elif pos < 128:
                base += max(0, 1.5 * (1 - pos / 128))

            # Recency boost
            dist_from_end = n - pos
            if dist_from_end <= 128:
                base += 2.0 + (128 - dist_from_end) * 0.02
            elif dist_from_end <= 256:
                base += 0.5

            # Sparse region penalty
            if self.rng.random() < 0.15:
                base -= 0.8

            block_scores.append(base)

        # Expand blocks to per-token scores with small jitter
        scores: list[float] = []
        for bi, bs in enumerate(block_scores):
            count = min(block, n - bi * block)
            for _ in range(count):
                scores.append(bs + self.rng.gauss(0, 0.15))

        scores = scores[:n]

        # Add surrogate noise based on tier R²
        r2 = {0: 0.50, 1: self.config.r2_linear, 2: self.config.r2_mlp}.get(
            surrogate_tier, self.config.r2_mlp
        )
        noise_std = math.sqrt(max(0, (1 - r2) / r2)) * base_sigma * 0.3
        if noise_std > 0:
            for i in range(len(scores)):
                scores[i] += self.rng.gauss(0, noise_std)

        # Compute stats
        global_mean = sum(scores) / len(scores)
        global_std = (sum((s - global_mean) ** 2 for s in scores) / len(scores)) ** 0.5

        # Approximate entropy from a small sample
        sample = scores[::max(1, len(scores) // 200)]
        min_s, max_s = min(sample), max(sample)
        nbins = 30
        entropy = 0.0
        if max_s > min_s and len(sample) > 1:
            bin_width = (max_s - min_s) / nbins
            bins = [0] * nbins
            for s in sample:
                b = min(int((s - min_s) / bin_width), nbins - 1)
                bins[b] += 1
            total = len(sample)
            entropy = -sum(
                (c / total) * math.log(c / total + 1e-10)
                for c in bins if c > 0
            )

        # Simplified per-layer stats
        per_layer_mean = [global_mean - 0.3 * (l / max(1, self.config.num_layers - 1))
                          for l in range(self.config.num_layers)]
        per_layer_std = [global_std] * self.config.num_layers

        return ScoreDistribution(
            scores=scores,
            seq_length=n,
            per_layer_mean=per_layer_mean,
            per_layer_std=per_layer_std,
            global_mean=global_mean,
            global_std=global_std,
            entropy=entropy,
        )
