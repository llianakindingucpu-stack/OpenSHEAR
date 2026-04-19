# OpenSHEAR Architecture

## Overview

OpenSHEAR (**S**tateless **H**ybrid **E**nsemble **A**rchitecture for **R**easoning) is a distributed AI inference framework where multiple independent **Cells** run small language models in parallel, and an **Aggregator** merges their outputs through competitive selection.

The key insight: instead of one massive model, run many small models independently and combine their strengths. This mirrors how the cerebral cortex works — ~10 billion cortical columns, each an independent processing unit, coordinated through sparse lateral connections.

```
┌──────────────────────────────────────────────────────┐
│                    Request Layer                       │
│  HTTP API / P2P Network / CLI                         │
└──────────────────────┬───────────────────────────────┘
                       │
┌──────────────────────▼───────────────────────────────┐
│                  Router / Credits                      │
│  Load balancing · Reputation tracking · Credit system  │
└──────────────────────┬───────────────────────────────┘
                       │
┌──────────────────────▼───────────────────────────────┐
│                   Aggregator                           │
│  WeightedSum · BestOfN · RankBased · Adaptive         │
└──┬──────────┬──────────┬──────────┬──────────────────┘
   │          │          │          │
┌──▼──┐  ┌───▼──┐  ┌───▼──┐  ┌───▼──┐
│Cell0│  │Cell1 │  │Cell2 │  │CellN │  ← All parallel
│200M │  │200M  │  │200M  │  │200M  │    No cross-talk
│state│  │state │  │state │  │state │    Local state only
└─────┘  └──────┘  └──────┘  └──────┘
```

## Core Principles (Invariant)

1. **No layer dependency** — Cells never feed into each other
2. **No global state** — No KV Cache synchronization, no shared hidden states
3. **Fully shardable** — Any Cell can be removed; the rest continue
4. **Preemption-safe** — Tasks can be reassigned; fastest result wins
5. **Minimal communication** — Each Cell output ≤ 512 bytes (token logits)

## Cell Architecture

Each Cell is a standalone small language model:

```
Input Token ──► Embedding ──► [TimeMix + FFN] × N_layers ──► LayerNorm ──► Output Logits
                              │           │
                              │           └── SwiGLU FFN (gated activation)
                              └── RWKV TimeMix (linear recurrence, O(1)/token)
```

### RWKV TimeMix (Linear Recurrence)

Unlike Transformers (O(n²) self-attention) or traditional RNNs (gated recurrence), TimeMix uses pure linear operations:

```
wkv = (decay * state + key * value) / (decay + key)
state_new = decay * state + key * value
```

- **O(1) per token** — no sequence-length dependency
- **Fixed-size state** — memory usage doesn't grow with context
- **No KV Cache** — eliminates the dominant memory bottleneck in Transformers
- **~3 matrix multiplications per timestep**

### SwiGLU FFN

```
output = (gate · σ(up)) · down
```

Where `σ` is the SiLU (Sigmoid Linear Unit) activation. This gated architecture has been shown to outperform standard ReLU/GELU FFNs in LLMs (PaLM, LLaMA).

### Cell State

Each Cell maintains a fixed-size local state (`CellState`) that does NOT grow with sequence length. This state is private to the Cell and never shared with other Cells.

```rust
pub struct CellState {
    pub time_mix_states: Vec<TimeMixState>,  // per-layer
    pub pos: usize,                          // current position
}

pub struct TimeMixState {
    pub aa: Vec<f32>,  // accumulator for attention [n_heads]
    pub bb: Vec<f32>,  // accumulator for attention [n_heads]
    pub pp: Vec<f32>,  // previous decay product [n_heads]
}
```

## Aggregator

The Aggregator merges N Cell outputs into final token probabilities. It is NOT a simple average — it uses competitive selection:

### Strategies

| Strategy | Formula | When to Use |
|----------|---------|-------------|
| **WeightedSum** | `Σ(w_i · logits_i)` | Default. Works well with diverse Cells. |
| **BestOfN** | `argmax_i(confidence_i) → logits_i` | When Cells have clear specializations. |
| **RankBased** | `Σ(rank(i) · logits_i)` | Smooth compromise between sum and best. |
| **Adaptive** | Dynamic strategy selection per-token | Best quality, slightly more compute. |

### Confidence Scoring

Each Cell's confidence is derived from its output distribution:

```
confidence_i = max(softmax(logits_i))
```

High confidence → the Cell is "sure" about its prediction. The Aggregator weights confident Cells more heavily.

## Speculative Decoding

SHEAR's architecture naturally enables speculative decoding — using faster Cells as drafters and slower (or more) Cells as verifiers:

```
┌──────────────────────────────────────────┐
│  Phase 1: Draft                          │
│  Cell 0 (low temp) ──► draft k tokens    │
│  Cell 1 (low temp) ──► draft k tokens    │
│  All cells run in parallel (rayon)        │
└──────────────────┬───────────────────────┘
                   │
┌──────────────────▼───────────────────────┐
│  Phase 2: Verify                         │
│  Main model verifies each drafted token   │
│  Accept ✓ or Reject ✗ + rollback         │
└──────────────────────────────────────────┘
```

### Current Implementation (Phase 1)

- **N Cells** with shared RWKV model weights (read-only)
- Each Cell has independent state + different sampling temperature
- **Per-token voting**: all Cells produce logits, vote on next token
- **Parallel execution** via rayon (`into_par_iter`)
- Consensus ratio controls acceptance threshold

### Future (Phase 2)

- **Cross-node speculative decoding**: L0/L1 nodes draft, L2/L3 nodes verify
- **k-token drafting**: generate k tokens before verification round
- **Heterogeneous models**: small model drafts, large model verifies

## Five-Level Node Hierarchy

The network supports heterogeneous hardware through a five-tier system:

| Level | Role | Hardware | Model Size | Function |
|-------|------|----------|------------|----------|
| **L0** | Collector | CPU only | None | Data collection, task routing |
| **L1** | Lightweight | CPU + 4GB RAM | 0.5B–1.5B | Draft generation, lightweight inference |
| **L2** | Standard | GPU (3060) | 7B | Standard inference, verification |
| **L3** | Heavy | GPU (3090/4090) | 14B+ | Heavy inference, training |
| **L4** | Datacenter | GPU (A100/H100) | 70B+ | Full training, dataset curation |

Each level contributes what it can. L0 nodes with only a CPU can still participate (data collection, routing). This is "not everyone equal, but everyone has a path."

## Credits & Reputation

- **Credits**: the unit of economic value. Earned by contributing compute, spent by consuming inference.
- **Reputation**: trust score (0–100). Increases with verified contributions, decreases with failed/incorrect results.
- **Verification**: three layers — requester annotation + redundant consensus + reputation-weighted scoring.

## Technology Stack

| Component | Technology | Reason |
|-----------|-----------|--------|
| Inference Engine | Rust (custom RWKV-4) | Performance, memory safety, no GC pauses |
| Parallelism | rayon | Zero-cost data parallelism |
| HTTP API | axum | Async, typed routing |
| P2P Network | tokio + custom protocol | Full-duplex, low-latency |
| Tokenizer | Custom BPE | Full merge-table support, no Python dependency |
| Database | SQLite (rusqlite) | Embedded, zero-config persistence |
| Weight Format | Custom binary | Direct mmap, no safetensors overhead at runtime |

## Comparison with Alternatives

| Feature | SHEAR | Petals | Distributed HF | Centralized API |
|---------|-------|--------|----------------|-----------------|
| Architecture | Parallel Cells | Pipeline shards | Single model | Single model |
| Communication | O(N) logits | O(L×d) activations | N/A | N/A |
| Fault tolerance | Cell removal OK | Shard failure = crash | N/A | N/A |
| Heterogeneous | ✓ (L0–L4) | ✗ (uniform) | ✗ | ✗ |
| Speculative Decoding | Built-in | ✗ | ✗ | ✗ |
| No KV Cache sync | ✓ | ✗ | N/A | N/A |
| CPU-only nodes | ✓ (L0/L1) | ✗ | ✗ | ✗ |
