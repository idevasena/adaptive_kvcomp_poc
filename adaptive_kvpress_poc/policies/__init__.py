"""Adaptive KV cache compression policies.

Implements three policy axes from the proposal:
  1. Threshold calibration (§3): adaptive τ selection
  2. Surrogate selection (§4): tiered scorer architecture choice
  3. Window policy (§5): adaptive sliding window sizing
"""

from .threshold import (
    FixedThresholdPolicy,
    AdaptiveThresholdPolicy,
    PercentileThresholdPolicy,
)
from .surrogate import SurrogateSelector, SurrogateTier
from .window import AdaptiveWindowPolicy, WindowConfig
from .compression import CompressionPolicy, CompressionResult

__all__ = [
    "FixedThresholdPolicy",
    "AdaptiveThresholdPolicy",
    "PercentileThresholdPolicy",
    "SurrogateSelector",
    "SurrogateTier",
    "AdaptiveWindowPolicy",
    "WindowConfig",
    "CompressionPolicy",
    "CompressionResult",
]
