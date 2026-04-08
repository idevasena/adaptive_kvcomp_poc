"""Adaptive KVPress PoC: trace-driven KV cache compression policy simulation.

An enhancement of KVZap/KVPress with adaptive threshold calibration,
surrogate selection, and window policy — evaluated using production
trace data from Mooncake (FAST'25) and NVIDIA Dynamo.

Usage:
    python run_simulation.py              # Full comparison on synthetic traces
    python run_simulation.py --fast       # Quick demo with fewer requests
"""

__version__ = "0.1.0"
