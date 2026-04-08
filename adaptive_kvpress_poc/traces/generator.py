"""Generate synthetic traces matching Mooncake / Azure production distributions.

These synthetic traces mirror the statistical properties reported in the
Mooncake FAST'25 paper and Dynamo benchmarking guide, including:
  - Log-normal input length distribution (mean ~8.7k, P99 ~62k)
  - Highly skewed block popularity (power-law)
  - Prefix reuse patterns (shared system prompts, multi-turn chat)
  - Realistic inter-arrival times (Poisson-like with burstiness)
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path

from .loader import MooncakeTrace, TraceEntry

BLOCK_SIZE = 512  # tokens per KV cache block (Mooncake convention)


def _generate_hash_ids(
    input_length: int,
    system_prompt_blocks: int,
    shared_prefix_pool: list[list[int]],
    block_counter: list[int],
    prefix_reuse_prob: float = 0.4,
) -> list[int]:
    """Generate block hash_ids with realistic prefix reuse structure.

    Simulates the Mooncake observation that:
      - System prompt blocks are shared across nearly all requests
      - ~50% of KV cache blocks can be reused (LRU hit ratio)
      - Block popularity follows a power-law distribution
    """
    num_blocks = max(1, (input_length + BLOCK_SIZE - 1) // BLOCK_SIZE)
    hash_ids: list[int] = []

    # 1. System prompt blocks (always reused)
    for i in range(min(system_prompt_blocks, num_blocks)):
        hash_ids.append(i)  # blocks 0..N are the system prompt

    remaining = num_blocks - len(hash_ids)
    if remaining <= 0:
        return hash_ids[:num_blocks]

    # 2. Possibly reuse a shared prefix from the pool
    if shared_prefix_pool and random.random() < prefix_reuse_prob:
        prefix = random.choice(shared_prefix_pool)
        reuse_len = min(len(prefix), remaining)
        hash_ids.extend(prefix[:reuse_len])
        remaining -= reuse_len

    # 3. Fill remainder with new unique blocks
    for _ in range(remaining):
        block_counter[0] += 1
        hash_ids.append(block_counter[0])

    # Possibly add this request's unique suffix to the shared pool
    if random.random() < 0.15 and len(hash_ids) > system_prompt_blocks + 2:
        suffix = hash_ids[system_prompt_blocks:]
        shared_prefix_pool.append(suffix)
        if len(shared_prefix_pool) > 200:
            shared_prefix_pool.pop(0)

    return hash_ids


def generate_synthetic_mooncake_trace(
    num_requests: int = 2000,
    duration_ms: int = 300_000,  # 5 minutes
    variant: str = "arxiv",
    seed: int = 42,
) -> MooncakeTrace:
    """Generate a synthetic trace matching Mooncake production statistics.

    Variants:
      - 'arxiv':        Long-context arXiv Q&A (original trace, mean ~8.7k tokens)
      - 'conversation': Multi-turn chat (shorter, high prefix reuse)
      - 'tool_agent':   Agentic/RAG workloads (moderate length, 50-80% block hit)
    """
    rng = random.Random(seed)
    random.seed(seed)

    # Distribution parameters per variant
    configs = {
        "arxiv": {
            "input_mu": 8.5, "input_sigma": 0.9,  # log-normal params → mean ~8.7k
            "output_mu": 4.5, "output_sigma": 0.8,
            "system_blocks": 3,
            "prefix_reuse_prob": 0.35,
            "max_input": 128_000,
        },
        "conversation": {
            "input_mu": 7.2, "input_sigma": 0.7,  # shorter, more concentrated
            "output_mu": 5.0, "output_sigma": 0.6,
            "system_blocks": 5,
            "prefix_reuse_prob": 0.65,  # high prefix reuse in chat
            "max_input": 32_000,
        },
        "tool_agent": {
            "input_mu": 7.8, "input_sigma": 1.0,
            "output_mu": 5.5, "output_sigma": 0.9,
            "system_blocks": 4,
            "prefix_reuse_prob": 0.55,
            "max_input": 64_000,
        },
    }
    cfg = configs.get(variant, configs["arxiv"])

    entries: list[TraceEntry] = []
    block_counter = [cfg["system_blocks"] + 100]  # start unique blocks above system
    shared_prefix_pool: list[list[int]] = []

    # Generate inter-arrival times (Poisson with burstiness)
    avg_interval = duration_ms / num_requests
    timestamps: list[float] = []
    t = 0.0
    for _ in range(num_requests):
        # Gamma-distributed inter-arrivals → bursty traffic
        interval = rng.gammavariate(2.0, avg_interval / 2.0)
        t += interval
        timestamps.append(t)

    # Normalize to fit within duration
    if timestamps:
        scale = duration_ms / timestamps[-1]
        timestamps = [t * scale for t in timestamps]

    for ts in timestamps:
        # Log-normal input length
        input_length = min(
            int(math.exp(rng.gauss(cfg["input_mu"], cfg["input_sigma"]))),
            cfg["max_input"],
        )
        input_length = max(64, input_length)  # minimum prompt

        # Log-normal output length
        output_length = max(1, int(math.exp(rng.gauss(cfg["output_mu"], cfg["output_sigma"]))))
        output_length = min(output_length, 8192)

        hash_ids = _generate_hash_ids(
            input_length,
            cfg["system_blocks"],
            shared_prefix_pool,
            block_counter,
            cfg["prefix_reuse_prob"],
        )

        entries.append(
            TraceEntry(
                timestamp=round(ts, 1),
                input_length=input_length,
                output_length=output_length,
                hash_ids=hash_ids,
            )
        )

    return MooncakeTrace(entries=entries, name=f"synthetic_{variant}")


def generate_synthetic_azure_trace(
    num_requests: int = 1000,
    duration_ms: int = 300_000,
    seed: int = 123,
) -> MooncakeTrace:
    """Generate a synthetic Azure LLM inference trace (no block hashes).

    Matches AzureLLMInferenceDataset2023/2024 distributions:
      - Generally shorter prompts than Mooncake
      - No prefix hash information
    """
    rng = random.Random(seed)
    entries: list[TraceEntry] = []

    avg_interval = duration_ms / num_requests
    t = 0.0
    for _ in range(num_requests):
        t += rng.expovariate(1.0 / avg_interval)
        input_length = max(32, min(32_000, int(math.exp(rng.gauss(7.0, 1.2)))))
        output_length = max(1, min(4096, int(math.exp(rng.gauss(4.0, 1.0)))))
        entries.append(
            TraceEntry(
                timestamp=round(t, 1),
                input_length=input_length,
                output_length=output_length,
                hash_ids=[],  # Azure traces don't include block hashes
            )
        )

    return MooncakeTrace(entries=entries, name="synthetic_azure")


def save_trace_jsonl(trace: MooncakeTrace, path: str | Path) -> None:
    """Save trace to Mooncake-compatible JSONL format."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for entry in trace:
            obj = {
                "timestamp": entry.timestamp,
                "input_length": entry.input_length,
                "output_length": entry.output_length,
                "hash_ids": entry.hash_ids,
            }
            f.write(json.dumps(obj) + "\n")
