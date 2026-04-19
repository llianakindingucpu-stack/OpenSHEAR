# Speculative Decoding Design

## Motivation

In a heterogeneous network, some nodes are fast (small model, CPU) and some are accurate (large model, GPU). Speculative decoding bridges this gap: fast nodes **draft** tokens, accurate nodes **verify** them.

SHEAR's architecture naturally supports this — Cells are already independent, so draft Cells and verify Cells can run on different hardware.

## Architecture

```
┌──────────────────────────────────────────────────────┐
│                    Draft Phase                         │
│                                                        │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐               │
│  │ Cell 0  │  │ Cell 1  │  │ Cell 2  │  ← L0/L1      │
│  │ T=0.5   │  │ T=0.8   │  │ T=1.1   │    nodes      │
│  └────┬────┘  └────┬────┘  └────┬────┘               │
│       │            │            │                      │
│       └────────────┼────────────┘                      │
│                    │                                    │
│            Ensemble Voting                             │
│            (Majority / Weighted)                       │
│                    │                                    │
│            draft_tokens = [t0, t1, ..., tk]           │
└────────────────────┬───────────────────────────────────┘
                     │
┌────────────────────▼───────────────────────────────────┐
│                  Verify Phase                            │
│                                                          │
│  ┌─────────────────────────────────────┐                │
│  │         Main Model (L2/L3)          │                │
│  │  For each drafted token:            │                │
│  │    if model agrees → accept ✓       │                │
│  │    else → reject ✗ + rollback       │                │
│  └─────────────────────────────────────┘                │
│                                                          │
│  accepted_tokens → output                                │
│  rejected → re-draft from last accepted position         │
└──────────────────────────────────────────────────────────┘
```

## Implementation Phases

### Phase 1: Single-Machine Ensemble Voting ✅ (Current)

- All Cells share the same RWKV model weights (read-only, no duplication)
- Each Cell has independent `RwkvModelState` + different sampling temperature
- Per-token voting: all Cells produce logits, vote on next token
- Parallel execution via rayon (`into_par_iter`)
- **Result**: 1.1x speedup over baseline (overhead eliminated by parallelism)

```rust
pub struct SpeculativeEngine {
    model: Arc<RwkvModel>,           // shared weights (read-only)
    states: Vec<RwkvModelState>,     // independent states per cell
    temperatures: Vec<f32>,          // diversity source
    strategy: VoteStrategy,          // Majority / ConfidenceWeighted / LeaderFollow
}
```

### Phase 2: Cross-Node Speculative Decoding (Design Complete)

- **L0/L1 nodes**: Draft cells (small model, fast CPU inference)
- **L2/L3 nodes**: Verifiers (large model, GPU-accelerated)
- Draft k tokens at once → verify batch → accept/reject
- **Rollback mechanism**: on rejection, reset verifier state to last accepted position

```
Draft (L0/L1, ~10 tok/s)      Verify (L2/L3, ~5 tok/s but higher quality)
  t0 t1 t2 t3 t4 ───────────► ✓ ✓ ✓ ✗ ──► accept t0-t2, re-draft from t2
  t3 t4 t5 ──────────────────► ✓ ✓ ✓ ──► accept t3-t5
```

Effective throughput: ~7-8 tok/s (vs. 5 tok/s without speculation) = **1.5-1.6x speedup**

### Phase 3: Heterogeneous Model Speculative Decoding (Future)

- Draft with small model (0.5B RWKV)
- Verify with large model (7B+ RWKV)
- Acceptance rate depends on draft-verify distribution alignment
- Theoretical speedup: **2-3x** (based on SpecMoE, Multi-Drafter papers)

## Voting Strategies

### Majority Vote
```
token = argmax_count(cells.map(|c| c.sample()))
```
Simple, robust, no confidence weighting needed.

### Confidence-Weighted
```
weights = softmax(cells.map(|c| max(softmax(c.logits))))
token = argmax(Σ(w_i · one_hot(sample_i)))
```
Cells with sharper distributions get more weight.

### Leader-Follow
```
leader_token = cells[0].sample()  // lowest temperature, most conservative
confirmations = cells[1..].filter(|c| c.sample() == leader_token).count()
if confirmations >= threshold { accept } else { re-sample }
```
Lowest-temperature Cell proposes, others confirm. Good for deterministic outputs.

## Consensus Tuning

The `min_consensus` parameter (0.0–1.0) controls how much agreement is required:

| min_consensus | Behavior | Trade-off |
|---------------|----------|-----------|
| 0.0 | Always accept majority | Fast, may be wrong |
| 0.33 | At least 1/3 agree (3 cells) | Balanced |
| 0.5 | At least majority agree | Default |
| 0.67 | At least 2/3 agree | Conservative |
| 1.0 | Unanimous agreement | Slow, but highest quality |

## Related Research

| Paper | Technique | SHEAR Application |
|-------|-----------|-------------------|
| SpecMoE (2026.04) | MoE-based speculative decoding | Cell = draft expert |
| Multi-Drafter | Multiple draft models | Multiple Cells draft in parallel |
| Fast Inference from Transformers | Speculative decoding theory | Draft-verify framework |
| TOPLOC | LSH-based output verification | Trust verification for draft tokens |
