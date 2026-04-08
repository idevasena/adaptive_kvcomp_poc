"""Mooncake / Dynamo trace data structures and JSONL loader."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator


@dataclass
class TraceEntry:
    """Single request from a production LLM inference trace.

    Matches the Mooncake open-source JSONL schema:
      {timestamp, input_length, output_length, hash_ids}

    The hash_ids are privacy-preserving 512-token block hashes that
    encode prefix-cache reuse structure without exposing content.
    """

    timestamp: float  # milliseconds from trace start
    input_length: int  # prompt tokens
    output_length: int  # generated tokens
    hash_ids: list[int] = field(default_factory=list)  # 512-tok block hashes

    @property
    def total_tokens(self) -> int:
        return self.input_length + self.output_length

    @property
    def num_kv_blocks(self) -> int:
        """Number of 512-token KV cache blocks."""
        return len(self.hash_ids) if self.hash_ids else (self.input_length + 511) // 512

    def prefix_overlap(self, other: "TraceEntry") -> int:
        """Count shared prefix blocks with another entry."""
        if not self.hash_ids or not other.hash_ids:
            return 0
        shared = 0
        for a, b in zip(self.hash_ids, other.hash_ids):
            if a != b:
                break
            shared += 1
        return shared


@dataclass
class MooncakeTrace:
    """A collection of trace entries with aggregate statistics."""

    entries: list[TraceEntry]
    name: str = "unnamed"

    # --- aggregate stats (computed lazily) ---------------------------------
    _block_frequency: dict[int, int] | None = field(default=None, repr=False)

    def __len__(self) -> int:
        return len(self.entries)

    def __iter__(self) -> Iterator[TraceEntry]:
        return iter(self.entries)

    def __getitem__(self, idx):
        return self.entries[idx]

    @property
    def duration_sec(self) -> float:
        if not self.entries:
            return 0.0
        return (self.entries[-1].timestamp - self.entries[0].timestamp) / 1000.0

    @property
    def mean_input_length(self) -> float:
        if not self.entries:
            return 0.0
        return sum(e.input_length for e in self.entries) / len(self.entries)

    @property
    def mean_output_length(self) -> float:
        if not self.entries:
            return 0.0
        return sum(e.output_length for e in self.entries) / len(self.entries)

    @property
    def block_frequency(self) -> dict[int, int]:
        """Frequency of each block hash across all requests."""
        if self._block_frequency is None:
            freq: dict[int, int] = {}
            for entry in self.entries:
                for h in entry.hash_ids:
                    freq[h] = freq.get(h, 0) + 1
            self._block_frequency = freq
        return self._block_frequency

    def summary(self) -> dict:
        """Return a summary dict for reporting."""
        lengths = [e.input_length for e in self.entries]
        lengths.sort()
        n = len(lengths)
        return {
            "name": self.name,
            "num_requests": n,
            "duration_sec": round(self.duration_sec, 1),
            "mean_input_tokens": round(self.mean_input_length, 1),
            "mean_output_tokens": round(self.mean_output_length, 1),
            "p50_input": lengths[n // 2] if n else 0,
            "p90_input": lengths[int(n * 0.9)] if n else 0,
            "p99_input": lengths[int(n * 0.99)] if n else 0,
            "max_input": lengths[-1] if n else 0,
            "unique_blocks": len(self.block_frequency),
            "total_block_refs": sum(self.block_frequency.values()),
        }


def load_mooncake_trace(path: str | Path, name: str | None = None) -> MooncakeTrace:
    """Load a Mooncake JSONL trace file.

    Expected format (one JSON object per line):
        {"timestamp": 27482, "input_length": 6955, "output_length": 52,
         "hash_ids": [46, 47, 48, ...]}
    """
    path = Path(path)
    entries: list[TraceEntry] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            entries.append(
                TraceEntry(
                    timestamp=obj["timestamp"],
                    input_length=obj["input_length"],
                    output_length=obj["output_length"],
                    hash_ids=obj.get("hash_ids", []),
                )
            )
    entries.sort(key=lambda e: e.timestamp)
    return MooncakeTrace(entries=entries, name=name or path.stem)
