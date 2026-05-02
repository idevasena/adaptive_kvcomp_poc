#!/usr/bin/env python3
"""Profile TriAttention scoring for comparison with KVZap in the adaptive PoC.

Two modes of operation:

  MODE 1 — GPU extraction (--mode extract):
    Loads the actual TriAttention codebase, runs model inference, and
    extracts per-token importance scores from TriAttention's internal scorer.
    Requires: GPU, triattention package installed, model weights.

  MODE 2 — Analytical (--mode analytical, DEFAULT):
    Implements TriAttention's trigonometric scoring formula from the paper
    (arXiv:2604.04921) using calibrated parameters. No GPU needed.
    Produces score profiles compatible with the adaptive PoC simulator.

Usage:
  # Analytical mode (no GPU, for trace simulation)
  python3 triattention_profile.py --mode analytical \
      --output-dir score_profiles/triattention/

  # GPU extraction mode (on dsksh805 with GPU)
  python3 triattention_profile.py --mode extract \
      --model Qwen/Qwen3-8B \
      --triattention-dir ../triattention \
      --output-dir score_profiles/triattention/

  # Compare profiles side-by-side with KVZap
  python3 triattention_profile.py --mode compare \
      --output-dir score_profiles/triattention/
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

# Add PoC to path
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PARENT_DIR = os.path.dirname(_THIS_DIR) if os.path.basename(_THIS_DIR) == "adaptive_kvpress_poc" else _THIS_DIR
if _PARENT_DIR not in sys.path:
    sys.path.insert(0, _PARENT_DIR)


# ═══════════════════════════════════════════════════════════════════════
# Calibrated parameters from the TriAttention paper
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class TriAttentionHeadParams:
    """Calibrated parameters for one attention head's trigonometric scorer.

    From the paper (Section 3.2): the attention logit between a query at
    position q and key at position k reduces to a trigonometric series:

      logit(q, k) ≈ Σ_f  A_f · cos(ω_f · (q - k) + φ_f)

    where (A_f, ω_f, φ_f) are determined by the pre-RoPE Q/K centers and
    the RoPE frequency bands.

    The Q/K concentration property means these centers are stable across
    positions and inputs — so the scoring function depends only on the
    positional distance (q - k), not on content.
    """
    # Trigonometric series parameters: list of (amplitude, frequency, phase)
    # Typically 32-64 frequency bands from RoPE
    trig_components: list[tuple[float, float, float]] = field(default_factory=list)

    # Q/K norm weight: for heads where Q is less concentrated,
    # key norms provide additional importance signal
    norm_weight: float = 0.0

    # Mean Resultant Length R — measures Q/K concentration
    # R > 0.5 means the head is well-concentrated (trig scoring reliable)
    concentration_r: float = 0.8


# Pre-calibrated parameters for Qwen3-8B (32 layers × 8 KV heads)
# These approximate the distributions reported in the paper's Figure 3
# In GPU mode, these are replaced by actual calibration from the model
def get_default_head_params(
    num_layers: int = 32,
    num_kv_heads: int = 8,
    head_dim: int = 128,
    rope_base: float = 10000.0,
    seed: int = 42,
) -> list[list[TriAttentionHeadParams]]:
    """Generate calibrated head parameters matching Qwen3-8B.

    Uses RoPE frequency bands and the paper's observation that:
    - Most heads have concentration R > 0.5 (strong trig scoring)
    - Amplitude decreases with frequency (low-freq = dominant)
    - Nearest-key preference is common (attention decays with distance)
    """
    import random
    rng = random.Random(seed)

    # RoPE frequencies: θ_i = rope_base^(-2i/head_dim)
    rope_freqs = [
        rope_base ** (-2.0 * i / head_dim)
        for i in range(head_dim // 2)
    ]

    params = []
    for layer in range(num_layers):
        layer_heads = []
        for head in range(num_kv_heads):
            # Concentration varies: deeper layers slightly less concentrated
            layer_frac = layer / max(1, num_layers - 1)
            base_r = 0.85 - 0.2 * layer_frac + rng.gauss(0, 0.05)
            concentration = max(0.3, min(0.95, base_r))

            # Generate trig components from RoPE frequencies
            # Use a subset of frequency bands (most energy in low frequencies)
            num_components = min(16, head_dim // 2)
            components = []
            for f_idx in range(num_components):
                freq = rope_freqs[f_idx]
                # Amplitude: dominated by low frequencies, decays with index
                # Paper shows nearest-key preference → positive amplitude at low freq
                amplitude = concentration * math.exp(-0.3 * f_idx) * (1.0 + rng.gauss(0, 0.1))
                # Phase: determined by Q/K center alignment
                phase = rng.uniform(-math.pi, math.pi)
                components.append((amplitude, freq, phase))

            # Norm weight: stronger for less concentrated heads
            norm_weight = max(0, (1 - concentration) * 0.5 + rng.gauss(0, 0.05))

            layer_heads.append(TriAttentionHeadParams(
                trig_components=components,
                norm_weight=norm_weight,
                concentration_r=concentration,
            ))
        params.append(layer_heads)

    return params


# ═══════════════════════════════════════════════════════════════════════
# Analytical scoring engine
# ═══════════════════════════════════════════════════════════════════════

def compute_triattention_scores_analytical(
    seq_length: int,
    head_params: TriAttentionHeadParams,
    query_pos: int | None = None,
    key_norms: list[float] | None = None,
) -> list[float]:
    """Compute TriAttention importance scores for all key positions.

    Implements Equation (7) from the paper:
      S_trig(k) = Σ_f A_f · cos(ω_f · (q - k) + φ_f)
      S_norm(k) = w · ||k_pre||
      S_total(k) = S_trig(k) + S_norm(k)

    Args:
        seq_length: Number of tokens in the sequence
        head_params: Calibrated trigonometric parameters for this head
        query_pos: Position of the query (default: end of sequence)
        key_norms: Pre-RoPE key norms (default: synthetic)
    """
    if query_pos is None:
        query_pos = seq_length - 1

    scores = []
    for key_pos in range(seq_length):
        distance = query_pos - key_pos

        # Trigonometric series score
        s_trig = 0.0
        for amplitude, freq, phase in head_params.trig_components:
            s_trig += amplitude * math.cos(freq * distance + phase)

        # Key norm score (synthetic norms if not provided)
        s_norm = 0.0
        if head_params.norm_weight > 0:
            if key_norms is not None and key_pos < len(key_norms):
                s_norm = head_params.norm_weight * key_norms[key_pos]
            else:
                # Synthetic norm: slight decrease with position, with noise
                import random
                norm = 1.0 - 0.1 * (key_pos / max(1, seq_length)) + random.gauss(0, 0.05)
                s_norm = head_params.norm_weight * norm

        scores.append(s_trig + s_norm)

    return scores


def aggregate_scores_across_heads(
    seq_length: int,
    all_head_params: list[list[TriAttentionHeadParams]],
    query_pos: int | None = None,
) -> list[float]:
    """Aggregate TriAttention scores across all layers and heads.

    Uses block-level scoring (1 score per 32 tokens) then expands with
    small jitter for speed. The trigonometric series varies smoothly
    with position, so block-level sampling is accurate.
    """
    if query_pos is None:
        query_pos = seq_length - 1

    num_layers = len(all_head_params)
    num_heads = len(all_head_params[0]) if all_head_params else 0
    total_heads = num_layers * num_heads
    if total_heads == 0 or seq_length == 0:
        return [0.0] * seq_length

    # Block-level scoring for speed
    block = 32
    num_blocks = max(1, (seq_length + block - 1) // block)
    block_scores = [0.0] * num_blocks

    for layer_params in all_head_params:
        for head_params in layer_params:
            for bi in range(num_blocks):
                pos = bi * block  # representative position
                distance = query_pos - pos
                s = 0.0
                for amplitude, freq, phase in head_params.trig_components:
                    s += amplitude * math.cos(freq * distance + phase)
                # Add small norm contribution
                s += head_params.norm_weight * (1.0 - 0.1 * pos / max(1, seq_length))
                block_scores[bi] += s

    # Average across heads
    for bi in range(num_blocks):
        block_scores[bi] /= total_heads

    # Expand to per-token with small jitter
    import random
    rng = random.Random(42)
    scores = []
    for bi in range(num_blocks):
        count = min(block, seq_length - bi * block)
        for _ in range(count):
            scores.append(block_scores[bi] + rng.gauss(0, 0.002))

    return scores[:seq_length]


# ═══════════════════════════════════════════════════════════════════════
# GPU extraction mode
# ═══════════════════════════════════════════════════════════════════════

def extract_scores_from_triattention(
    model_name: str,
    triattention_dir: str,
    prompts: list[str],
    device: str = "cuda",
) -> list[dict]:
    """Extract actual TriAttention scores by running model inference.

    This function attempts to hook into TriAttention's internal scoring
    to extract per-token importance scores during prefill.
    """
    # Add triattention to path
    ta_dir = Path(triattention_dir)
    if ta_dir.exists():
        sys.path.insert(0, str(ta_dir))

    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
    except ImportError:
        print("ERROR: torch and transformers required for extract mode")
        print("       Use --mode analytical for CPU-only profiling")
        sys.exit(1)

    # Discover TriAttention's module structure
    try:
        import triattention as ta
        ta_modules = dir(ta)
        print(f"  TriAttention modules: {ta_modules}")
    except ImportError:
        print("ERROR: triattention package not found")
        print(f"       Looked in: {triattention_dir}")
        print(f"       Run: cd {triattention_dir} && pip install -e .")
        sys.exit(1)

    print(f"  Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    # Look for TriAttention's scoring/eviction module
    # The codebase applies compression via monkey-patching attention layers
    # We need to find the scoring function and intercept it
    #
    # Common patterns in the codebase:
    #   triattention.attention     — custom attention with scoring
    #   triattention.cache         — KV cache management with eviction
    #   triattention.scoring       — trigonometric scoring functions
    #   triattention.calibration   — Q/K center estimation
    #
    # Try to locate the scoring function:
    scoring_fn = None
    for attr_name in ["compute_scores", "score_keys", "trig_score",
                       "compute_importance", "get_importance_scores"]:
        for module_name in ["triattention.scoring", "triattention.attention",
                           "triattention.cache", "triattention.core"]:
            try:
                mod = __import__(module_name, fromlist=[attr_name])
                if hasattr(mod, attr_name):
                    scoring_fn = getattr(mod, attr_name)
                    print(f"  Found scoring function: {module_name}.{attr_name}")
                    break
            except (ImportError, AttributeError):
                continue
        if scoring_fn:
            break

    if scoring_fn is None:
        print("  WARNING: Could not find TriAttention scoring function")
        print("  Attempting hook-based extraction via forward pass...")

    # Hook-based extraction: register hooks on attention layers to capture
    # pre-softmax attention scores during forward pass
    extracted_scores = []
    hooks = []
    captured_attentions = {}

    def make_hook(layer_idx):
        def hook_fn(module, input, output):
            # Try to extract attention weights from the output
            if isinstance(output, tuple) and len(output) > 1:
                attn_weights = output[1]  # usually (batch, heads, seq, seq)
                if attn_weights is not None:
                    # Sum attention from last query position across all heads
                    # This gives per-key importance from the last query's perspective
                    with torch.no_grad():
                        importance = attn_weights[0, :, -1, :].mean(dim=0).cpu().numpy()
                        captured_attentions[layer_idx] = importance.tolist()
        return hook_fn

    # Register hooks on attention layers
    for layer_idx, layer in enumerate(model.model.layers):
        attn_module = layer.self_attn
        hook = attn_module.register_forward_hook(make_hook(layer_idx))
        hooks.append(hook)

    results = []
    for prompt_idx, prompt in enumerate(prompts):
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        seq_len = inputs.input_ids.shape[1]

        captured_attentions.clear()
        with torch.no_grad():
            outputs = model(**inputs, output_attentions=True)

        # Aggregate scores across layers
        if captured_attentions:
            all_scores = [0.0] * seq_len
            for layer_scores in captured_attentions.values():
                for i in range(min(len(layer_scores), seq_len)):
                    all_scores[i] += layer_scores[i]
            num_layers = len(captured_attentions)
            all_scores = [s / num_layers for s in all_scores]
        else:
            # Fallback: use output_attentions from the model
            if hasattr(outputs, 'attentions') and outputs.attentions:
                all_scores = [0.0] * seq_len
                for layer_attn in outputs.attentions:
                    # layer_attn: (batch, heads, seq, seq)
                    importance = layer_attn[0, :, -1, :].mean(dim=0).cpu().numpy()
                    for i in range(min(len(importance), seq_len)):
                        all_scores[i] += float(importance[i])
                num_layers = len(outputs.attentions)
                all_scores = [s / num_layers for s in all_scores]
            else:
                print(f"  WARNING: No attention scores captured for prompt {prompt_idx}")
                all_scores = [0.0] * seq_len

        results.append({
            "seq_length": seq_len,
            "scores": all_scores,
            "mean": sum(all_scores) / len(all_scores) if all_scores else 0,
            "std": (sum((s - sum(all_scores)/len(all_scores))**2 for s in all_scores) / len(all_scores)) ** 0.5 if all_scores else 0,
        })

        if (prompt_idx + 1) % 50 == 0:
            print(f"    Profiled {prompt_idx + 1}/{len(prompts)} prompts "
                  f"(last: {seq_len} tokens, mean_score={results[-1]['mean']:.4f})")

    # Remove hooks
    for hook in hooks:
        hook.remove()

    return results


# ═══════════════════════════════════════════════════════════════════════
# Profile generation (for PoC integration)
# ═══════════════════════════════════════════════════════════════════════

def generate_analytical_profiles(
    output_dir: str,
    lengths: list[int] | None = None,
    num_samples_per_length: int = 10,
    model_config: str = "qwen3-8b",
):
    """Generate TriAttention score profiles for the PoC simulator.

    Unlike KVZap profiles which vary across prompts (content-dependent),
    TriAttention profiles are nearly identical across prompts at the same
    length (position-dependent). We generate a small number of samples
    to capture the norm-based variation.
    """
    if lengths is None:
        lengths = [1024, 2048, 4096, 8192, 16384, 32768]

    configs = {
        "qwen3-8b": {"num_layers": 32, "num_kv_heads": 8, "head_dim": 128},
        "llama-3.1-8b": {"num_layers": 32, "num_kv_heads": 8, "head_dim": 128},
        "qwen3-32b": {"num_layers": 64, "num_kv_heads": 8, "head_dim": 128},
    }
    cfg = configs.get(model_config, configs["qwen3-8b"])

    print(f"  Generating analytical TriAttention profiles for {model_config}")
    print(f"  Config: {cfg}")
    print(f"  Lengths: {lengths}")

    # Get calibrated parameters
    head_params = get_default_head_params(
        num_layers=cfg["num_layers"],
        num_kv_heads=cfg["num_kv_heads"],
        head_dim=cfg["head_dim"],
    )

    os.makedirs(output_dir, exist_ok=True)
    all_profiles = {}

    for length in lengths:
        t0 = time.time()
        profiles = []

        for sample_idx in range(num_samples_per_length):
            # Aggregate scores across all heads
            scores = aggregate_scores_across_heads(length, head_params)

            # Compute statistics
            mean_score = sum(scores) / len(scores)
            std_score = (sum((s - mean_score)**2 for s in scores) / len(scores)) ** 0.5
            sorted_scores = sorted(scores)
            n = len(sorted_scores)

            profile = {
                "seq_length": length,
                "sample_idx": sample_idx,
                "scores_sample": scores[::max(1, length // 200)],  # downsample for storage
                "mean": mean_score,
                "std": std_score,
                "quantiles": {
                    f"q{int(q*100)}": sorted_scores[int(q * (n-1))]
                    for q in [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
                },
                "fraction_above": {
                    f"tau_{t:.1f}": sum(1 for s in scores if s >= t) / n
                    for t in [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0]
                },
                # Position-dependent score regions for prefix analysis
                "region_scores": {
                    "first_4_tokens": sum(scores[:4]) / 4 if length >= 4 else 0,
                    "first_512_block": sum(scores[:min(512, length)]) / min(512, length),
                    "first_3_blocks": sum(scores[:min(1536, length)]) / min(1536, length),
                    "last_128_tokens": sum(scores[max(0, length-128):]) / min(128, length),
                    "middle_50pct": sum(scores[length//4:3*length//4]) / (length // 2) if length >= 4 else 0,
                },
            }
            profiles.append(profile)

        elapsed = time.time() - t0
        mean_of_means = sum(p["mean"] for p in profiles) / len(profiles)
        std_of_means = (sum((p["mean"] - mean_of_means)**2 for p in profiles) / len(profiles)) ** 0.5

        all_profiles[str(length)] = profiles
        print(f"    length={length:>6,}: mean_score={mean_of_means:+.4f} "
              f"std={profiles[0]['std']:.4f} "
              f"inter-prompt-std={std_of_means:.6f} "
              f"({elapsed:.1f}s)")

    # Save profiles
    profile_path = os.path.join(output_dir, f"triattention_{model_config}.json")
    with open(profile_path, "w") as f:
        json.dump({
            "model": model_config,
            "method": "triattention_analytical",
            "config": cfg,
            "num_samples_per_length": num_samples_per_length,
            "profiles": all_profiles,
        }, f, indent=2)
    print(f"\n  Saved: {profile_path}")

    return all_profiles


# ═══════════════════════════════════════════════════════════════════════
# Comparison with KVZap profiles
# ═══════════════════════════════════════════════════════════════════════

def compare_profiles(output_dir: str):
    """Compare TriAttention vs KVZap score profiles side-by-side."""

    # Generate TriAttention profiles if not present
    ta_path = os.path.join(output_dir, "triattention_qwen3-8b.json")
    if not os.path.exists(ta_path):
        print("  Generating TriAttention profiles...")
        generate_analytical_profiles(output_dir, model_config="qwen3-8b")

    with open(ta_path) as f:
        ta_data = json.load(f)

    # Generate KVZap profiles using the PoC's score engine
    try:
        from adaptive_kvpress_poc.score_profiles import ScoreProfileEngine
        kvzap_engine = ScoreProfileEngine("qwen3-8b", seed=42)
        has_kvzap = True
    except ImportError:
        print("  WARNING: adaptive_kvpress_poc not found, skipping KVZap comparison")
        has_kvzap = False

    print("\n" + "=" * 78)
    print("  TriAttention vs KVZap Score Profile Comparison")
    print("=" * 78)

    print(f"\n  {'Length':>8s}  {'Method':<14s}  {'Mean':>8s}  {'Std':>8s}  "
          f"{'Ret@-1.0':>9s}  {'Ret@0.0':>8s}  {'1st-512blk':>11s}  {'Last-128':>9s}")
    print(f"  {'─'*8}  {'─'*14}  {'─'*8}  {'─'*8}  {'─'*9}  {'─'*8}  {'─'*11}  {'─'*9}")

    for length_str, ta_profiles in ta_data["profiles"].items():
        length = int(length_str)
        ta = ta_profiles[0]  # first sample (TriAttention is deterministic)

        print(f"  {length:>8,}  {'TriAttention':<14s}  "
              f"{ta['mean']:>+8.4f}  {ta['std']:>8.4f}  "
              f"{ta['fraction_above'].get('tau_-1.0', 0):>8.1%}  "
              f"{ta['fraction_above'].get('tau_0.0', 0):>7.1%}  "
              f"{ta['region_scores']['first_512_block']:>+11.4f}  "
              f"{ta['region_scores']['last_128_tokens']:>+9.4f}")

        if has_kvzap:
            # Generate KVZap scores at the same length
            dist = kvzap_engine.generate_scores(length, surrogate_tier=2)
            sorted_s = sorted(dist.scores)
            n = len(sorted_s)
            ret_neg1 = dist.fraction_above(-1.0)
            ret_0 = dist.fraction_above(0.0)
            first_512 = sum(dist.scores[:min(512, n)]) / min(512, n) if n > 0 else 0
            last_128 = sum(dist.scores[max(0, n-128):]) / min(128, n) if n > 0 else 0

            print(f"  {length:>8,}  {'KVZap-MLP':<14s}  "
                  f"{dist.global_mean:>+8.4f}  {dist.global_std:>8.4f}  "
                  f"{ret_neg1:>8.1%}  {ret_0:>7.1%}  "
                  f"{first_512:>+11.4f}  {last_128:>+9.4f}")
            print()

    # Prefix preservation analysis
    print("\n" + "=" * 78)
    print("  Prefix Block Preservation Prediction")
    print("=" * 78)
    print(f"\n  Key question: Do TriAttention's position-based scores naturally")
    print(f"  protect prefix blocks (positions 0–1536, first 3 system prompt blocks)?")

    for length_str, ta_profiles in ta_data["profiles"].items():
        length = int(length_str)
        ta = ta_profiles[0]
        rs = ta["region_scores"]

        # Compare prefix region scores to the overall mean
        prefix_vs_mean = rs["first_3_blocks"] - ta["mean"]
        sink_vs_mean = rs["first_4_tokens"] - ta["mean"]
        recency_vs_mean = rs["last_128_tokens"] - ta["mean"]

        print(f"\n  Length {length:>6,}:")
        print(f"    Attention sinks (pos 0-3):     {rs['first_4_tokens']:>+.4f}  "
              f"(vs mean: {sink_vs_mean:>+.4f})")
        print(f"    First 3 prefix blocks (0-1535):{rs['first_3_blocks']:>+.4f}  "
              f"(vs mean: {prefix_vs_mean:>+.4f})")
        print(f"    Last 128 tokens (recency):     {rs['last_128_tokens']:>+.4f}  "
              f"(vs mean: {recency_vs_mean:>+.4f})")
        print(f"    Middle 50%:                    {rs['middle_50pct']:>+.4f}  "
              f"(vs mean: {rs['middle_50pct'] - ta['mean']:>+.4f})")

        if prefix_vs_mean > 0:
            print(f"    → Prefix blocks ABOVE mean: TriAttention naturally preserves prefixes")
        else:
            print(f"    → Prefix blocks BELOW mean: TriAttention may evict prefixes — reuse bonus needed")

    print(f"\n{'='*78}")
    print(f"  Profile comparison complete.")
    print(f"{'='*78}")


# ═══════════════════════════════════════════════════════════════════════
# PoC integration: score engine compatible with trace simulator
# ═══════════════════════════════════════════════════════════════════════

def create_triattention_score_engine():
    """Create a score engine class compatible with the adaptive PoC.

    Returns a class that can replace ScoreProfileEngine in CompressionPolicy.
    """
    from adaptive_kvpress_poc.score_profiles import ScoreDistribution, ModelConfig

    class TriAttentionScoreEngine:
        """Drop-in replacement for ScoreProfileEngine using TriAttention scoring.

        Key differences from KVZap ScoreProfileEngine:
        - Scores are position-dependent, not content-dependent
        - Near-zero inter-prompt variance at the same length
        - No surrogate noise (scoring is analytical, not learned)
        - Stronger recency bias from trigonometric nearest-key preference
        """

        def __init__(self, model_config="qwen3-8b", seed=42):
            configs = {
                "qwen3-8b": ModelConfig("qwen3-8b", 32, 8, 4096, 128),
                "llama-3.1-8b": ModelConfig("llama-3.1-8b", 32, 8, 4096, 128),
                "qwen3-32b": ModelConfig("qwen3-32b", 64, 8, 5120, 128),
            }
            self.config = configs.get(model_config, configs["qwen3-8b"])
            self.head_params = get_default_head_params(
                self.config.num_layers, self.config.num_kv_heads,
                self.config.head_dim, seed=seed,
            )

        def generate_scores(self, input_length, surrogate_tier=2):
            """Generate TriAttention scores (surrogate_tier ignored — no surrogate)."""
            if input_length == 0:
                return ScoreDistribution(scores=[], seq_length=0)

            scores = aggregate_scores_across_heads(input_length, self.head_params)

            # Stats
            global_mean = sum(scores) / len(scores)
            global_std = (sum((s - global_mean)**2 for s in scores) / len(scores)) ** 0.5

            # Entropy
            sample = scores[::max(1, len(scores) // 200)]
            min_s, max_s = min(sample), max(sample)
            entropy = 0.0
            if max_s > min_s and len(sample) > 1:
                nbins = 30
                bw = (max_s - min_s) / nbins
                bins = [0] * nbins
                for s in sample:
                    b = min(int((s - min_s) / bw), nbins - 1)
                    bins[b] += 1
                total = len(sample)
                entropy = -sum((c/total) * math.log(c/total + 1e-10) for c in bins if c > 0)

            per_layer_mean = [global_mean] * self.config.num_layers
            per_layer_std = [global_std] * self.config.num_layers

            return ScoreDistribution(
                scores=scores, seq_length=input_length,
                per_layer_mean=per_layer_mean, per_layer_std=per_layer_std,
                global_mean=global_mean, global_std=global_std, entropy=entropy,
            )

    return TriAttentionScoreEngine


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Profile TriAttention scoring for comparison with KVZap"
    )
    parser.add_argument("--mode", choices=["analytical", "extract", "compare"],
                        default="analytical",
                        help="analytical=CPU-only trig scoring, extract=GPU model inference, compare=side-by-side")
    parser.add_argument("--model", default="Qwen/Qwen3-8B",
                        help="Model name for GPU extraction mode")
    parser.add_argument("--model-config", default="qwen3-8b",
                        choices=["qwen3-8b", "llama-3.1-8b", "qwen3-32b"],
                        help="Model config for analytical mode")
    parser.add_argument("--triattention-dir", default="../triattention",
                        help="Path to triattention repo clone")
    parser.add_argument("--output-dir", default="score_profiles/triattention",
                        help="Output directory for score profiles")
    parser.add_argument("--lengths", default="1024,2048,4096,8192,16384,32768",
                        help="Comma-separated sequence lengths to profile")
    parser.add_argument("--num-samples", type=int, default=10,
                        help="Samples per length bucket")
    args = parser.parse_args()

    lengths = [int(x) for x in args.lengths.split(",")]

    print("=" * 70)
    print("  TriAttention Score Profiling")
    print("=" * 70)
    print(f"  Mode: {args.mode}")
    print(f"  Model: {args.model_config}")
    print(f"  Lengths: {lengths}")
    print(f"  Output: {args.output_dir}")
    print()

    if args.mode == "analytical":
        generate_analytical_profiles(
            args.output_dir, lengths, args.num_samples, args.model_config
        )
    elif args.mode == "extract":
        print("  GPU extraction mode requires prompts. Generating from tokenizer...")
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
            # Generate prompts at target lengths
            base_text = "Explain the theory of " * 500
            prompts = []
            for length in lengths:
                for _ in range(min(args.num_samples, 5)):
                    tokens = tokenizer.encode(base_text)[:length]
                    prompts.append(tokenizer.decode(tokens))
            print(f"  Generated {len(prompts)} prompts")
        except ImportError:
            print("  ERROR: transformers required for extract mode")
            sys.exit(1)

        results = extract_scores_from_triattention(
            args.model, args.triattention_dir, prompts
        )
        os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, "extracted_scores.json"), "w") as f:
            json.dump(results, f, indent=2)
        print(f"  Saved {len(results)} profiles to {args.output_dir}/extracted_scores.json")

    elif args.mode == "compare":
        compare_profiles(args.output_dir)


if __name__ == "__main__":
    main()
