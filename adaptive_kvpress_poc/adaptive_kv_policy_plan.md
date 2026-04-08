# Adaptive KV Cache Compression Policy: Threshold Calibration, Surrogate Selection & Window Policy

## Plan Based on KVzap (arXiv:2601.07891) and NVIDIA kvpress Framework

---

## 1. Executive Summary

KVzap demonstrates that a small surrogate model (linear or 2-layer MLP) trained to approximate KVzip+ oracle scores can achieve 2–4× KV cache compression with negligible accuracy loss. The key insight is that **threshold-based pruning** rather than fixed top-k selection makes compression inherently input-adaptive — the same threshold τ yields 20% variation in compression ratio across prompts based on information density. This plan proposes extending KVzap's approach into a fully adaptive policy that dynamically calibrates thresholds, selects surrogates, and adjusts window size based on workload characteristics, with implementation grounded in the kvpress framework.

**Evaluation approach**: Rather than running full LLM inference for calibration and ablations, this plan uses **production trace data from Mooncake (Kimi/Moonshot AI) and NVIDIA Dynamo** to drive offline simulation. Mooncake's open-sourced traces (23,608 production requests with timestamps, token lengths, and prefix-cache block hashes) combined with a one-time score profiling step (~500 prompts, single GPU) enable rapid iteration on compression policies without GPU-intensive inference loops. End-to-end validation with Dynamo's AIPerf benchmarking tooling is reserved as a final confirmation step.

---

## 2. Analysis of KVzap Results & Key Observations

### 2.1 Surrogate Fidelity vs. Accuracy

KVzap trains per-layer surrogates to regress hidden states → log KVzip+ scores. The squared Pearson correlation (R²) between predicted and oracle scores ranges from **0.63 to 0.77**, with MLP consistently outperforming Linear. Despite imperfect correlation, the ranking fidelity is sufficient for pruning because we only need correct ordering at the threshold boundary, not exact score prediction.

**Critical observation**: The gap between Linear (R² ≈ 0.63) and MLP (R² ≈ 0.77) translates to roughly 0.5–1.0 points on RULER at equivalent compression. This means the surrogate architecture choice has a measurable but bounded effect — suggesting there's a regime where the cheaper Linear surrogate is "good enough" and another where the MLP is necessary.

### 2.2 Threshold Behavior Across Benchmarks

The paper reports that a single fixed threshold τ yields:
- **RULER (synthetic, repetitive)**: High compression ratios (>0.7 removed at 4k/16k)
- **LongBench (real-world, dense)**: Lower compression ratios at the same τ
- **AIME25 (math reasoning)**: Moderate compression, with MLP maintaining pass@4 near 2× compression

This confirms that threshold-based pruning is naturally adaptive to information density, but the threshold itself is currently set statically per model. A fixed τ optimized for RULER will over-compress dense documents; one optimized for LongBench will under-compress synthetic/repetitive content.

### 2.3 Window Policy Observations

KVzap uses a fixed sliding window of **w = 128 most recent tokens**, following StreamingLLM conventions. During decoding, scores of window tokens are set to ∞ (always retained). This is effective but leaves room for improvement:
- Some tasks (multi-hop QA, summarization) may need larger windows for coherent local context
- Reasoning tasks with long chains-of-thought may benefit from adaptive windows that grow during critical inference steps
- 128 tokens is arbitrary and not justified by any workload-specific analysis

### 2.4 The kvpress Integration Landscape

The kvpress framework provides several relevant building blocks:
- **`DMSPress`** (Dynamic Minimum Score): evicts below a threshold rather than top-k — this is the threshold-based wrapper KVzap uses
- **`KVzapPress`**: the surrogate scorer, to be used with DMSPress
- **`AdaKVPress`**: head-wise adaptive budget allocation
- **`PerLayerCompressionPress`**: per-layer compression ratios
- **`CriticalKVPress`**: two-stage refinement using value norms
- **`ComposedPress`**: chaining multiple presses
- **`PrefillDecodingPress`**: separate strategies for prefill vs. decoding phases
- **`DecodingPress`**: periodic compression during token generation

---

## 3. Adaptive Threshold Calibration

### 3.1 Problem Statement

A single fixed threshold τ cannot optimally serve all input distributions. The goal is to dynamically select τ such that a **target compression quality** (measured by downstream accuracy proxy) is maintained, rather than a target compression ratio.

### 3.2 Proposed Approach: Score Distribution Profiling

**Phase 1 — Offline Calibration Curves**

For each model (e.g., Qwen3-8B, Llama-3.1-8B), build calibration curves that map:
- τ → mean compression ratio on representative prompt categories (code, natural language, math, multilingual)
- τ → accuracy retention on held-out validation tasks

Implementation sketch:
```python
# Pseudocode for offline calibration
for category in ["code", "natural_language", "math", "multilingual"]:
    for tau in np.linspace(-2.0, 2.0, 50):
        compression_ratios = []
        for prompt in calibration_set[category]:
            scores = kvzap_model(hidden_states)
            retained = (scores >= tau).sum() / scores.numel()
            compression_ratios.append(1.0 / retained)
        calibration_curves[category][tau] = {
            "mean_compression": np.mean(compression_ratios),
            "std_compression": np.std(compression_ratios),
        }
```

**Phase 2 — Online Score Distribution Classifier**

At inference time, compute lightweight statistics from the KVzap score distribution *before* pruning:
- Mean, variance, and skewness of predicted scores across all layers/heads
- Entropy of the score distribution (high entropy → diverse importance → need conservative τ)
- Fraction of scores above multiple threshold candidates

Use these statistics to classify the prompt into one of the offline-profiled categories and select the corresponding calibrated τ.

**Phase 3 — Feedback-Driven τ Adjustment (Decoding Phase)**

During decoding with `DecodingPress`, monitor for signs of quality degradation:
- Perplexity spike on the most recent generation window
- Attention entropy collapse (model "can't find" what it needs)
- Token repetition rate increase

If degradation signals exceed a threshold, reduce τ (retain more KV pairs) for subsequent compression intervals.

### 3.3 Implementation in kvpress

Extend `DMSPress` to support a callable threshold:

```python
class AdaptiveDMSPress(DMSPress):
    """DMSPress with dynamic threshold selection."""
    
    def __init__(self, base_press, threshold_fn, fallback_threshold=-1.0):
        super().__init__(base_press=base_press, threshold=fallback_threshold)
        self.threshold_fn = threshold_fn
    
    def score(self, module, hidden_states, keys, values, attentions, kwargs):
        scores = self.base_press.score(module, hidden_states, keys, values, attentions, kwargs)
        # Compute score statistics for adaptive threshold
        score_stats = {
            "mean": scores.mean().item(),
            "std": scores.std().item(),
            "entropy": -(scores.softmax(-1) * scores.log_softmax(-1)).sum(-1).mean().item(),
        }
        self.threshold = self.threshold_fn(score_stats)
        return scores
```

### 3.4 Calibration Storage

Store calibration tables as lightweight JSON artifacts per model checkpoint, versioned alongside the KVzap surrogate weights. These can be loaded at initialization and require no GPU compute.

---

## 4. Surrogate Selection Strategy

### 4.1 Problem Statement

KVzap offers two surrogate architectures — Linear (0.02% overhead) and MLP (1.1% overhead). The paper treats this as a fixed choice, but an adaptive system should pick the right surrogate for the situation.

### 4.2 Proposed Multi-Tier Surrogate Architecture

**Tier 0 — No Surrogate (Heuristic Fallback)**

For extremely latency-sensitive workloads or when surrogate weights aren't available, fall back to a training-free scorer from kvpress (e.g., `KnormPress` or `ExpectedAttentionPress`). These are already competitive at moderate compression ratios (≤2×).

**Tier 1 — Linear Surrogate (KVzap-Linear)**

Use when:
- Target compression ratio ≤ 2×
- Prompt length > 64k tokens (where the marginal FLOP cost matters most)
- Batch serving with tight latency SLOs
- R² ≥ 0.6 on the model's validation set is sufficient

**Tier 2 — MLP Surrogate (KVzap-MLP)**

Use when:
- Target compression ratio > 2× (aggressive pruning needs better score ranking)
- Task requires high faithfulness (reasoning, code generation, multi-hop QA)
- R² improvement of MLP over Linear exceeds 0.1 for the specific model/layer

**Tier 3 — Ensemble / Critical Refinement**

For maximum quality at cost of latency:
- Use KVzap-MLP scores refined by `CriticalKVPress` (L1 norm of Wo @ values)
- Or ensemble KVzap scores with a complementary scorer like `LeverageScorePress`

### 4.3 Automatic Tier Selection

```python
def select_surrogate_tier(
    target_compression: float,
    sequence_length: int,
    task_type: str,  # "retrieval", "reasoning", "generation", "summarization"
    latency_budget_ms: float,
) -> int:
    if target_compression <= 1.5:
        return 0  # heuristic is enough
    
    if task_type in ("reasoning", "code") and target_compression > 2.0:
        return 2  # need MLP for aggressive + faithful
    
    if sequence_length > 65536 and latency_budget_ms < 50:
        return 1  # linear for latency-critical long contexts
    
    if target_compression > 2.5:
        return 2  # MLP for aggressive compression
    
    return 1  # linear as default
```

### 4.4 Per-Layer Surrogate Mixing

An advanced strategy: not all layers need the same surrogate quality. The paper shows R² varies significantly by layer. For layers where Linear achieves R² > 0.7, keep Linear; for layers where MLP gains > 0.15 R², use MLP. This creates a **hybrid stack** that saves FLOPs on easy layers while spending compute on hard ones.

Implementation via `PerLayerCompressionPress` combined with layer-specific surrogate loading.

---

## 5. Adaptive Window Policy

### 5.1 Problem Statement

The fixed 128-token sliding window is a blunt instrument. Different tasks, different generation phases, and different compression ratios call for different amounts of guaranteed local context.

### 5.2 Proposed Adaptive Window Strategies

**Strategy A — Compression-Ratio-Proportional Window**

As compression becomes more aggressive, increase the window to compensate for lost global context:

```python
def adaptive_window_size(compression_ratio: float, base_window: int = 128) -> int:
    """Scale window with compression aggressiveness."""
    if compression_ratio <= 2.0:
        return base_window
    elif compression_ratio <= 4.0:
        return int(base_window * (1 + 0.5 * (compression_ratio - 2.0)))
        # 2× → 128, 3× → 192, 4× → 256
    else:
        return min(base_window * 3, 512)  # cap at 512
```

**Strategy B — Task-Aware Window Sizing**

Different task families have different locality requirements:

| Task Type | Recommended Window | Rationale |
|-----------|-------------------|-----------|
| Single-hop QA | 128 (base) | Answer locality is high |
| Multi-hop QA | 256–384 | Need to track reasoning chain |
| Summarization | 128 | Compression itself acts as summarization |
| Code generation | 256 | Function-level context matters |
| Math reasoning (CoT) | 384–512 | Long chains of dependent steps |
| Chat/dialogue | 192 | Recent turns need preservation |

**Strategy C — Score-Aware Dynamic Window**

Instead of a fixed recent-token window, make the window boundary score-aware: retain the most recent W tokens *plus* any tokens in the boundary region whose KVzap score exceeds a secondary threshold τ_local > τ_global. This prevents hard cutoffs from evicting locally important tokens that happen to fall just outside the window.

```python
def score_aware_window(scores, positions, window_size=128, tau_local_factor=0.8):
    """Retain window tokens + high-scoring near-boundary tokens."""
    seq_len = scores.shape[-1]
    window_start = max(0, seq_len - window_size)
    
    # Always keep the window
    keep_mask = positions >= window_start
    
    # Also keep high-scoring tokens in the boundary region
    boundary_start = max(0, window_start - 64)
    boundary_mask = (positions >= boundary_start) & (positions < window_start)
    tau_local = scores[keep_mask].min() * tau_local_factor
    boundary_keep = boundary_mask & (scores > tau_local)
    
    return keep_mask | boundary_keep
```

**Strategy D — Attention Sink Preservation**

Following StreamingLLM's insight that initial tokens act as attention sinks, the window policy should also include a **head window** of the first K tokens (typically 4–8). KVzap implicitly handles this if those tokens have high scores, but explicitly reserving them adds robustness.

### 5.3 Unified Window Policy

Combine strategies into a single configurable policy:

```python
@dataclass
class AdaptiveWindowPolicy:
    base_window: int = 128
    head_tokens: int = 4  # attention sink tokens
    scale_with_compression: bool = True
    score_aware_boundary: bool = True
    boundary_extension: int = 64
    task_type: str = "general"
    
    def compute_keep_mask(self, scores, compression_ratio, seq_len):
        # 1. Head tokens (attention sinks)
        head_mask = torch.zeros(seq_len, dtype=torch.bool)
        head_mask[:self.head_tokens] = True
        
        # 2. Adaptive tail window
        w = self.base_window
        if self.scale_with_compression:
            w = adaptive_window_size(compression_ratio, w)
        w = min(w, TASK_WINDOWS.get(self.task_type, w))
        
        tail_mask = torch.zeros(seq_len, dtype=torch.bool)
        tail_mask[max(0, seq_len - w):] = True
        
        # 3. Score-aware boundary extension
        boundary_mask = torch.zeros(seq_len, dtype=torch.bool)
        if self.score_aware_boundary:
            boundary_mask = score_aware_window(
                scores, torch.arange(seq_len), w, tau_local_factor=0.8
            )
        
        return head_mask | tail_mask | boundary_mask
```

---

## 6. Integrated Adaptive Policy Architecture

### 6.1 End-to-End Flow

```
Input Prompt
    │
    ▼
┌─────────────────────────┐
│  Score Distribution      │ ← KVzap surrogate (tier selected by §4)
│  Profiling               │
└──────────┬──────────────┘
           │ score stats (mean, std, entropy)
           ▼
┌─────────────────────────┐
│  Adaptive Threshold      │ ← Calibration lookup (§3)
│  Selection               │   Returns τ_adaptive
└──────────┬──────────────┘
           │ τ_adaptive
           ▼
┌─────────────────────────┐
│  Window Policy           │ ← Task-aware + compression-scaled (§5)
│  Computation             │   Returns keep_mask
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│  KV Cache Pruning        │ ← scores > τ OR keep_mask → retain
│  (prefill & decoding)    │
└─────────────────────────┘
```

### 6.2 kvpress Composition

```python
from kvpress import (
    KVzapPress, DMSPress, AdaKVPress, CriticalKVPress,
    ComposedPress, PrefillDecodingPress, DecodingPress
)

# Build the adaptive press
kvzap = KVzapPress(model_name="qwen3-8b", variant="mlp")

# Wrap with adaptive threshold
adaptive_dms = AdaptiveDMSPress(
    base_press=kvzap,
    threshold_fn=calibrated_threshold_lookup,
    window_policy=AdaptiveWindowPolicy(
        base_window=128,
        head_tokens=4,
        scale_with_compression=True,
        task_type="reasoning",
    ),
)

# Optional: head-wise adaptive budget
adaptive_press = AdaKVPress(adaptive_dms)

# Combined prefill + decoding strategy
full_press = PrefillDecodingPress(
    prefilling_press=adaptive_press,
    decoding_press=DecodingPress(
        base_press=kvzap,
        compression_steps=10,
        token_buffer_size=256,
    ),
)
```

### 6.3 Configuration Profiles

Define named profiles for common deployment scenarios:

| Profile | Surrogate | τ Strategy | Window | Target Compression |
|---------|-----------|------------|--------|-------------------|
| `conservative` | MLP | percentile-based (p10) | 256 + sinks | 1.5–2× |
| `balanced` | MLP | calibrated mean | 128 + sinks | 2–3× |
| `aggressive` | MLP + Critical | calibrated + feedback | 192 + score-aware | 3–4× |
| `latency-optimal` | Linear | fixed low τ | 128 | 2× |
| `memory-optimal` | MLP | calibrated aggressive | 128 | 4×+ |

---

## 7. Trace-Driven Evaluation: Using Dynamo & Mooncake Footprint Data

A key design goal of this plan is to **avoid running full LLM inference** for calibration and most evaluation steps. Instead, we leverage production-grade trace datasets from NVIDIA Dynamo and Mooncake (Moonshot AI / KVCache.ai) to drive offline simulation of compression policies. This approach is faster, cheaper, reproducible, and tests against real workload distributions rather than synthetic benchmarks.

### 7.1 Available Trace Datasets

#### Mooncake Production Traces

The Mooncake project (FAST'25, serving Kimi) open-sources real inference traces from production arXiv workloads as JSONL files at `github.com/kvcache-ai/Mooncake`. Three trace variants exist:

| Trace Variant | Characteristics | Best For |
|---------------|-----------------|----------|
| **arxiv** (original) | 23,608 requests over 60 min; mean 8,772 tokens; P99 ~62k tokens | General workload simulation, prefix reuse analysis |
| **conversation** | Multi-turn chat patterns; high prefix reuse in recent KVC blocks | Chat/dialogue window policy testing |
| **tool & agent** | Agentic workloads; KVC block hit ratio >50–80% | RAG/agent compression policy |

Each trace entry contains:
```json
{
  "timestamp": 27482,
  "input_length": 6955,
  "output_length": 52,
  "hash_ids": [46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 2353, 2354]
}
```

The `hash_ids` are privacy-preserving 512-token block hashes that enable prefix cache reuse analysis without exposing user content. Block popularity is highly skewed — over 50% of blocks are unused while popular blocks (system prompts) are accessed thousands of times.

#### Dynamo KV Smart Router Benchmark Traces

NVIDIA Dynamo's A/B benchmarking guide uses the Mooncake arxiv trace for KV-cache-aware routing evaluation. The Dynamo documentation provides ready-made tooling to replay these traces against inference backends with configurable request timing (fixed-schedule replay) or throughput stress-testing. The NVIDIA AIPerf profiler can directly ingest Mooncake traces via `--custom-dataset-type mooncake_trace`.

#### Azure LLM Inference Traces (Supplementary)

Microsoft's `Azure/AzurePublicDataset` provides complementary traces:
- **AzureLLMInferenceDataset2023**: One-day sample from two Azure LLM services (from Splitwise/ISCA'24)
- **AzureLLMInferenceDataset2024**: One-week sample (from DynamoLLM/HPCA'25)

These contain input/output token counts and arrival timestamps but not block hashes. Useful for validating token-length distribution assumptions across providers.

### 7.2 Trace-Driven Simulation Architecture

Instead of running LLM inference, we build a **compression policy simulator** that processes trace entries through a pre-computed score model:

```
Trace Entry (input_length, output_length, hash_ids)
    │
    ▼
┌─────────────────────────────────┐
│  Score Distribution Generator    │ ← Synthetic score distributions
│  (parameterized by input_length, │   fitted from a small set of
│   hash_id reuse pattern,         │   actual KVzap runs (one-time cost)
│   and block position)            │
└──────────┬──────────────────────┘
           │ simulated per-token scores
           ▼
┌─────────────────────────────────┐
│  Adaptive Policy Under Test      │ ← Threshold calibration (§3)
│  (threshold τ, window w,         │   Surrogate tier selection (§4)
│   surrogate tier)                │   Window policy (§5)
└──────────┬──────────────────────┘
           │ decisions: retain/evict per block
           ▼
┌─────────────────────────────────┐
│  Metrics Collector               │ ← Compression ratio achieved
│                                  │   Memory footprint over time
│                                  │   Prefix reuse efficiency
│                                  │   Estimated TTFT/TBT impact
└─────────────────────────────────┘
```

#### One-Time Score Profiling (The Only LLM Step)

Run KVzap scoring on a **small representative sample** (~500 prompts across length buckets from Nemotron) to build score distribution templates:

```python
# One-time profiling: ~500 prompts, single GPU, a few hours
score_profiles = {}
for length_bucket in [1024, 2048, 4096, 8192, 16384, 32768]:
    prompts = sample_nemotron(n=50, min_len=length_bucket*0.8, max_len=length_bucket*1.2)
    for prompt in prompts:
        hidden_states = model.prefill(prompt)
        scores = kvzap_model(hidden_states)  # per-layer, per-head scores
        score_profiles[length_bucket].append({
            "mean": scores.mean(dim=-1),       # per-layer means
            "std": scores.std(dim=-1),
            "quantiles": torch.quantile(scores, q=torch.linspace(0,1,21), dim=-1),
            "fraction_above": [(scores > tau).float().mean() for tau in thresholds],
        })
```

These profiles are stored as compact numpy arrays (~few MB per model) and used to generate synthetic score distributions for any trace entry based on its `input_length`.

#### Trace Replay Without LLM

For each Mooncake trace entry, the simulator:

1. **Maps `input_length` → score distribution template** from the one-time profiling
2. **Maps `hash_ids` → prefix reuse pattern** — shared hash_id prefixes indicate tokens that could be served from KV cache rather than recomputed, affecting the effective sequence length for compression decisions
3. **Applies the adaptive policy** (threshold selection, window sizing, surrogate tier) to the synthetic scores
4. **Records compression outcomes**: tokens retained, memory footprint, prefix blocks reusable, estimated TTFT savings

### 7.3 Metrics (Trace-Driven)

| Metric | Source | How Computed |
|--------|--------|-------------|
| **Compression ratio distribution** | Mooncake trace × policy | Tokens retained / total per request, across all 23k requests |
| **Memory footprint timeline** | Mooncake trace timestamps | KV cache memory over the 60-min trace with eviction + new arrivals |
| **Prefix reuse efficiency** | hash_ids overlap analysis | How often compressed cache blocks remain useful for subsequent requests with shared prefixes |
| **Threshold stability** | Fixed τ vs. adaptive τ | CV of compression ratio across requests at same quality target |
| **Policy overhead estimate** | Score profile FLOPs | Surrogate tier FLOPs / total attention FLOPs per request |
| **SLO compliance (estimated)** | TTFT model from Mooncake paper | Prefill time prediction using input_length − prefix_hit_length, factoring in compression savings |

### 7.4 Ablation Studies (All Trace-Driven)

All ablations below run purely on trace data + pre-computed score profiles — **no GPU inference needed**:

- **Fixed vs. adaptive threshold**: Replay full Mooncake arxiv trace with τ ∈ {-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0} vs. adaptive τ(score_stats). Compare CV of compression ratio.
- **Window size sweep**: 64, 128, 192, 256, 384, 512 tokens with and without head-token sinks. Measure how many prefix-reusable blocks fall inside vs. outside the window.
- **Score-aware boundary extension**: Hard window vs. score-aware. Quantify additional retained tokens and prefix reuse gain.
- **Surrogate tier selection**: For each trace entry, simulate Linear vs. MLP vs. heuristic decisions using the R² profiles from the KVzap paper. Estimate accuracy impact via the calibrated score→accuracy curves.
- **Cross-trace generalization**: Train calibration on arxiv trace, evaluate on conversation and tool&agent traces. Tests whether threshold curves transfer across workload types.
- **Mooncake vs. Azure traces**: Validate that token length distributions and compression ratio patterns are consistent across the two production trace sources.

### 7.5 Dynamo Integration Testing

For end-to-end validation with actual inference (Phase 5 only), use Dynamo's benchmarking infrastructure:

```bash
# Replay Mooncake trace through Dynamo with adaptive compression enabled
# Uses Dynamo's KV Smart Router + KVBM with custom compression hook
aiperf profile \
    --model qwen3-8b \
    --url $DYNAMO_ENDPOINT \
    --endpoint-type chat \
    --streaming \
    --input-file mooncake_arxiv_trace.jsonl \
    --custom-dataset-type mooncake_trace \
    --fixed-schedule \
    --tokenizer Qwen/Qwen3-8B
```

This validates that the trace-simulated compression decisions match actual inference behavior, but is only needed as a final confirmation step — not for the iterative development loop.

### 7.6 Prefix Reuse × Compression Interaction Analysis

A unique advantage of the Mooncake traces is the `hash_ids` field, which enables studying how KV cache compression interacts with prefix caching — a dimension the KVzap paper doesn't explore. Specifically:

- **Question**: If we compress a KV cache block before storing it in the Dynamo KVBM or Mooncake Store, does the compressed version remain useful for subsequent requests that share the same prefix?
- **Analysis**: For each pair of requests sharing hash_id prefixes, compare: (a) the set of tokens retained by compression in the first request, vs. (b) the tokens that the second request would need from the shared prefix. If compression evicts tokens that the second request needs, prefix reuse is broken.
- **Implication for window policy**: Tokens in frequently-reused prefix blocks (high hash_id reuse count) should receive a score bonus, making them less likely to be evicted. This creates a **reuse-aware compression policy** that goes beyond KVzap's per-request optimization.

```python
def reuse_aware_score_bonus(scores, block_positions, block_reuse_counts):
    """Boost scores of tokens in frequently-reused prefix blocks."""
    bonus = torch.zeros_like(scores)
    for block_idx, (start, end) in enumerate(block_positions):
        reuse_count = block_reuse_counts[block_idx]
        if reuse_count > 1:
            # Log-scaled bonus: more reuse → higher retention priority
            bonus[..., start:end] = math.log1p(reuse_count) * 0.5
    return scores + bonus
```

---

## 8. Implementation Roadmap

### Phase 1: Score Profiling & Trace Infrastructure (1–2 weeks)
- Run one-time KVzap scoring on ~500 Nemotron prompts across length buckets (single GPU, few hours)
- Store score distribution profiles as compact numpy archives per model
- Download and parse Mooncake traces (arxiv, conversation, tool&agent variants)
- Download Azure LLM Inference traces as supplementary validation
- Build trace replay simulator that maps trace entries → synthetic score distributions

### Phase 2: Adaptive Threshold in kvpress (2 weeks)
- Extend `DMSPress` with callable threshold support
- Implement `AdaptiveDMSPress` with score-distribution profiling
- Add calibration lookup integration
- **Validate on Mooncake trace**: run full 23k-request simulation, compare fixed vs. adaptive τ

### Phase 3: Window Policy Extensions (1–2 weeks)
- Implement `AdaptiveWindowPolicy` dataclass
- Add attention sink preservation to KVzap's sliding window logic
- Add score-aware boundary extension
- **Validate on Mooncake trace**: measure prefix reuse efficiency across window configurations

### Phase 4: Surrogate Selection & Reuse-Aware Policy (1–2 weeks)
- Implement tier selection logic
- Per-layer hybrid surrogate loading
- Fallback to training-free scorers
- Implement reuse-aware score bonus using hash_id frequency from Mooncake traces
- Run cross-trace generalization study (arxiv → conversation → tool&agent)

### Phase 5: End-to-End Validation with Dynamo (2 weeks)
- Deploy Qwen3-8B with kvpress + adaptive policy on Dynamo (aggregated or disaggregated)
- Replay Mooncake trace via AIPerf with `--fixed-schedule` for realistic timing
- Compare trace-simulated predictions vs. actual TTFT/TBT/compression measurements
- Validate prefix reuse efficiency with Dynamo KVBM tiered storage
- Final profile-tuning for named deployment configurations

---

## 9. Open Questions & Risks

1. **Score profile fidelity**: The one-time profiling on ~500 Nemotron prompts creates score distribution templates parameterized by length. If production workloads (Mooncake's arXiv, conversation, agent traces) have fundamentally different score distributions (e.g., due to domain-specific content structure), the synthetic scores may not capture real compression behavior. Mitigation: validate score profiles against a small sample of actual KVzap runs on representative prompts from each trace category.

2. **Hash_id granularity mismatch**: Mooncake traces use 512-token block hashes, but KVzap operates at individual token granularity. The simulator must bridge this gap — either by assuming uniform importance within a 512-token block or by applying position-dependent score shaping within blocks. The choice will affect prefix reuse efficiency estimates.

3. **Prefix reuse invalidation under compression**: When compressed KV cache blocks are stored in Dynamo KVBM or Mooncake Store, the block hash remains the same but the content is sparser. Subsequent requests matching the same hash_id may need tokens that were evicted. This is a novel failure mode not studied in either the KVzap or Mooncake papers. The reuse-aware score bonus (§7.6) mitigates this, but the right bonus scaling needs empirical tuning.

4. **Feedback latency during decoding**: Perplexity-based τ adjustment during decoding adds monitoring overhead. The Mooncake trace timestamps allow us to simulate whether feedback delay (e.g., adjusting τ every N tokens) causes SLO violations under realistic request arrival rates.

5. **Cross-provider trace differences**: Mooncake traces come from Kimi's arXiv-heavy workloads; Azure traces cover different services. Token length distributions differ significantly (Mooncake P99 ~62k tokens vs. Azure which may be shorter). Calibration curves should be validated on both to ensure robustness.

6. **Interaction with quantization**: How does adaptive thresholding interact with NVFP4 or INT4 KV cache quantization? Score distributions may shift under quantization — the one-time profiling should include both FP16 and quantized runs. Dynamo's NVFP4 KV cache support provides a natural testbed.

7. **Scaling to larger models**: KVzap results cover 8B and 32B. The Mooncake production system serves models at much larger scale (LLaMA2-70B architecture). Score profiles and threshold curves for 70B+ models need separate one-time profiling runs.

---

## 10. References

- Jégou & Jeblick, "KVzap: Fast, Adaptive, and Faithful KV Cache Pruning," arXiv:2601.07891, Jan 2026
- NVIDIA kvpress framework: github.com/NVIDIA/kvpress
- Qin et al., "Mooncake: Trading More Storage for Less Computation — A KVCache-centric Architecture for Serving LLM Chatbot," FAST'25 / USENIX, 2025. Traces: github.com/kvcache-ai/Mooncake
- NVIDIA Dynamo: github.com/ai-dynamo/dynamo. KV Smart Router A/B benchmarking guide uses Mooncake traces.
- Microsoft Azure LLM Inference Traces: github.com/Azure/AzurePublicDataset (AzureLLMInferenceDataset2023, 2024)
- Kim et al., "KVzip: Query-Agnostic KV Cache Compression with Context Reconstruction," NeurIPS 2025
- Devoto et al., "Expected Attention," arXiv:2510.00636, 2025
- Xiao et al., "Efficient Streaming Language Models with Attention Sinks," ICLR 2024
- Chari & Van Durme, "Compactor: Calibrated KV Cache Compression with Approximate Leverage Scores," arXiv:2507.08143, 2025
- Xu et al., "Towards Efficient Key-Value Cache Management for Prefix LLM Serving," arXiv:2505.21919, 2025 (Mooncake trace analysis)
