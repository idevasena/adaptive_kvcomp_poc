"""Adaptive window policy for KV cache sliding window management.

KVzap uses a fixed 128-token sliding window. This module provides:
  - Compression-ratio-proportional window scaling
  - Task-aware window sizing
  - Score-aware boundary extension
  - Attention sink (head token) preservation
"""

from __future__ import annotations

import math
from dataclasses import dataclass


# Task-specific recommended window sizes
TASK_WINDOWS = {
    "single_hop_qa": 128,
    "multi_hop_qa": 320,
    "summarization": 128,
    "code_generation": 256,
    "math_reasoning": 448,
    "chat_dialogue": 192,
    "general": 128,
    "agent": 256,
}


@dataclass
class WindowConfig:
    """Configuration for the adaptive window policy."""

    base_window: int = 128
    head_tokens: int = 4  # attention sink tokens
    scale_with_compression: bool = True
    score_aware_boundary: bool = True
    boundary_extension: int = 64
    task_type: str = "general"
    max_window: int = 512


class AdaptiveWindowPolicy:
    """Computes which token positions are protected from eviction."""

    def __init__(self, config: WindowConfig | None = None):
        self.config = config or WindowConfig()

    def compute_keep_mask(
        self,
        seq_length: int,
        compression_ratio: float,
        scores: list[float] | None = None,
    ) -> list[bool]:
        """Compute per-token keep mask (True = protected from eviction).

        Protected regions:
          1. Head tokens (attention sinks): positions 0..head_tokens
          2. Tail window: last W tokens (W adapts to compression + task)
          3. Score-aware boundary: high-scoring tokens near window edge
        """
        cfg = self.config
        mask = [False] * seq_length

        if seq_length == 0:
            return mask

        # 1. Head tokens (attention sinks)
        for i in range(min(cfg.head_tokens, seq_length)):
            mask[i] = True

        # 2. Adaptive tail window
        w = self._compute_window_size(compression_ratio)
        tail_start = max(0, seq_length - w)
        for i in range(tail_start, seq_length):
            mask[i] = True

        # 3. Score-aware boundary extension
        if cfg.score_aware_boundary and scores is not None:
            boundary_start = max(0, tail_start - cfg.boundary_extension)
            if tail_start > 0 and tail_start < seq_length:
                # Threshold: min score in the window (tokens we're already keeping)
                window_scores = [scores[i] for i in range(tail_start, seq_length)
                                 if i < len(scores)]
                if window_scores:
                    tau_local = min(window_scores) * 0.8
                    for i in range(boundary_start, tail_start):
                        if i < len(scores) and scores[i] > tau_local:
                            mask[i] = True

        return mask

    def _compute_window_size(self, compression_ratio: float) -> int:
        """Scale window size with compression aggressiveness."""
        cfg = self.config
        w = cfg.base_window

        # Task-aware base
        task_w = TASK_WINDOWS.get(cfg.task_type, cfg.base_window)
        w = max(w, task_w)

        # Compression-ratio scaling
        if cfg.scale_with_compression and compression_ratio > 2.0:
            scale = 1.0 + 0.5 * (compression_ratio - 2.0)
            w = int(w * scale)

        return min(w, cfg.max_window)

    def get_effective_window_size(self, compression_ratio: float) -> int:
        """Public accessor for the computed window size."""
        return self._compute_window_size(compression_ratio)

    @property
    def name(self) -> str:
        parts = [f"window(base={self.config.base_window}"]
        if self.config.head_tokens > 0:
            parts.append(f"sinks={self.config.head_tokens}")
        if self.config.scale_with_compression:
            parts.append("scaled")
        if self.config.score_aware_boundary:
            parts.append("score-aware")
        parts.append(f"task={self.config.task_type})")
        return ",".join(parts)
