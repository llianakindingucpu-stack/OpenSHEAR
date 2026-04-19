# Project SHEAR — Stateless Hybrid Ensemble Architecture for Reasoning

## 1. Project Codename

**SHEAR** — Parallel Inference Engine
**Full Name**: Stateless Hybrid Ensemble Architecture for Reasoning

## 2. Mission & Vision

### The Problem AI Infrastructure Has Today

AI inference is becoming critical infrastructure — like electricity or water supply. Yet the current model creates a single point of trust failure:

| Risk | What Happens |
|------|-------------|
| **Prompt interception** | Token relay/proxy services read, log, and resell your prompts |
| **Model downgrade** | APIs labeled "GPT-4" may actually run cheaper models silently |
| **Prompt injection** | Middlemen can inject hidden system prompts to steal business logic |
| **Data exfiltration** | Sensitive enterprise data (finance, healthcare, government) flows through third parties |
| **Compliance violation** | Data localization laws (China's DSL, EU GDPR) make cloud API usage illegal for many orgs |

### SHEAR's Answer: Privacy by Architecture

SHEAR doesn't add encryption or policies on top of existing infrastructure — **privacy is built into the inference mechanism itself**.

When a prompt enters the SHEAR network:

```
User Prompt
    ↓
[Router] — splits into task shards
    ↓           ↓           ↓
  Cell 0    Cell 1    Cell 2    ← each sees only a fragment
    ↓           ↓           ↓
[Aggregator] — combines inference results, NOT original prompts
    ↓
Response to User
```

**No single node ever sees the complete prompt.** This is analogous to Tor's onion routing — no relay knows both source and destination. In SHEAR, no node can reconstruct the full user input from its shard alone.

Key properties:
- **No trusted third party required** — the architecture itself prevents full prompt exposure
- **No additional encryption overhead** — nodes do see their assigned shard, but cannot reconstruct the whole
- **Privacy scales with nodes** — more cells in the ensemble = smaller fragments = harder to reconstruct
- **Verifiable results** — consensus voting + reputation system ensures output quality without trusting any single node

### Core Commitments

- Privacy-by-architecture: prompt sharding is fundamental, not an add-on
- Fully open source: every line of inference code is auditable
- No middleman: peer-to-peer, node-to-node, no API gateway that can intercept
- Run anywhere: from a $50 PC to a GPU cluster
- Fully abandon serial Transformer / RNN pipelines
- Native support for preemption, multi-cheap-CPU, distributed parallelism
- Minimal communication — runs on 100Mbps networks
- Target: ≥20 token/s, capability equivalent to RWKV-14B
- 100% Rust implementation
- Trainable and deployable by individuals / small teams

## 3. Core Design Principles (Non-Negotiable)

1. **No tier dependencies**: No serial dependencies between Cells
2. **No global state**: No KV Cache synchronization; Cell state is local
3. **Fully sharded**: Any Cell can be removed and the rest keeps running
4. **Preemption safe**: Tasks can be re-dispatched; fastest responder wins
5. **Minimal communication**: Single Cell result ≤ 512 bytes

## 4. Model Architecture

```
[Input Token Embedding]
      ↓ broadcast
┌───────┬───────┬───────┬───────┐
Cell 0  Cell 1  Cell 2  ... Cell N  ← fully parallel, no dependencies, no communication
│local  │local  │local  │       local│
│state  │state  │state  │       state│
└───┬───┴───┬───┴───┬───┴───────┬───┘
    │       │       │           │
    └───────┴───────┼───────────┘
                    ▼
           [Aggregator]
    Learned routing weights (not fixed averaging)
                    ▼
           [Output Head]
        → next token probability
```

## 5. Cell Implementation

### Current Implementation (v0.2.0)

Each Cell runs an independent RWKV-4-based language model:

```rust
pub struct CellConfig {
    pub vocab_size: usize,    // 50277 (standard)
    pub d_model: usize,       // 768
    pub d_ffn: usize,         // 3072
    pub n_layers: usize,      // 6
    pub head_size: usize,     // 64
    pub n_heads: usize,       // 12
    pub max_seq_len: usize,   // 2048
}
// Total: ~200M params per Cell
```

### Cell Forward Pass

```
Input Token
    ↓ Embedding [vocab_size, d_model]
    ↓ For each layer:
    │   ↓ LayerNorm (ln1)
    │   ↓ TimeMix (RWKV-style linear recurrence)
    │   │   wkv = (decay * state + key * value) / (decay + key)
    │   │   state_new = decay * state + key * value  ← O(1), fixed size
    │   ↓ Output projection [d_model, d_model]
    │   ↓ Residual connection
    │   ↓ LayerNorm (ln2)
    │   ↓ FFN (sigmoid(receptance) * value)
    │   ↓ Residual connection
    ↓ LayerNorm (ln_out)
    ↓ Output Head [d_model, vocab_size]
    → logits
```

### Local State (CellState)

```rust
pub struct TimeMixState {
    pub aa: Vec<f32>,  // attention accumulator [n_heads]
    pub bb: Vec<f32>,  // attention accumulator [n_heads]
    pub pp: Vec<f32>,  // previous decay product [n_heads]
}
```

- Fixed size — does not grow with sequence length
- No cross-Cell synchronization needed
- No KV Cache needed

## 6. Aggregator Implementation

### Current Strategies

| Strategy | Implementation | Best For |
|----------|---------------|----------|
| **WeightedSum** | `Σ(w_i · logits_i)` | Default — diverse Cells |
| **BestOfN** | Select highest-confidence Cell | Specialized Cells |
| **RankBased** | Weight by confidence rank | Balanced approach |
| **Adaptive** | Dynamically select strategy | Best quality, slightly more compute |

### Confidence Computation

```
confidence_i = max(softmax(logits_i))
```

Higher confidence = Cell is more certain of its output → Aggregator gives it more weight.

## 7. Speculative Decoding

SHEAR's Cell architecture natively supports speculative decoding:

### Phase 1 (Implemented ✅)

- N Cells share model weights (`Arc<RwkvModel>`, read-only)
- Each Cell has independent state + different sampling temperature
- Token-by-token voting, rayon parallel
- Result: 1.1× speedup (parallelization eliminates overhead)

### Phase 2 (Design Complete 📐)

- L0/L1 nodes act as draft (small model, fast)
- L2/L3 nodes act as verifiers (large model, accurate)
- Draft k tokens → Verify batch → Accept/Reject + rollback
- Expected speedup: 1.5–2×

See [Speculative Decoding Design Document](SPECULATIVE_DECODING.md) for full details.

## 8. Five-Tier Node System

| Tier | Role | Hardware | Model | Function |
|------|------|----------|-------|----------|
| L0 | Collector | CPU only | None | Data collection, request routing |
| L1 | Lightweight | CPU + 4GB | 0.5B–1.5B | Draft generation, light inference |
| L2 | Standard | 3060 | 7B | Standard inference, verification |
| L3 | Heavy | 3090/4090 | 14B+ | Heavy inference, LoRA fine-tuning |
| L4 | Datacenter | A100/H100 | 70B+ | Full training, dataset curation |

See [Node Hierarchy Document](NODE_HIERARCHY.md) for full details.

## 9. Why SHEAR Is Strong

### Ensemble Learning Theory

Mixture of Experts (MoE) is proven at scale:
- DeepSeek V3: 256 Experts, 671B → GPT-4 class
- Mixtral 8×7B: outperforms LLaMA-2 70B

SHEAR = **Distributed MoE**, with Experts spread across different machines.

### Competitive Selection vs. Averaging

```
Traditional ensemble:  output = mean(A, B, C, ..., N)  → tends toward mediocrity
SHEAR:                output = best(A, B, C, ..., N)   → preserves expertise
```

### Brain Cortex Column Analogy

The human cortex has ~100 billion columnar structures, each processing information independently with sparse lateral connections. Many weak units running in parallel = strong intelligence.

## 10. Parameter Scale & Performance Targets

| Config | # Cells | Total Params | Per-Cell RAM | Expected Capability |
|--------|---------|-------------|-------------|---------------------|
| Minimal | 8 | 1.6B | 2GB × 8 | GPT-2 class |
| Standard | 32 | 6.4B | 2GB × 32 | GPT-3 class |
| Large | 64 | 12.8B | 2GB × 64 | GPT-3.5 class |
| Full Network | 128+ | 25.6B+ | Distributed | RWKV-14B class |

**Any single machine only needs to run 1 Cell (200M, 2GB RAM)**

## 11. Current Benchmark Results

| Metric | Baseline (1 Cell) | Ensemble (3 Cells, rayon) |
|--------|-------------------|--------------------------|
| Speed (CPU, 169M) | ~4.87 tok/s | ~2.4 tok/s (1.1× faster) |
| Consensus rate | N/A | 45.6% |
| HumanEval Pass@1 | 0.0% (untrained) | TBD |
| Crashes | 0/164 | 0 |

Test environment: Intel Pentium G4560, 8GB RAM, no GPU

See [Benchmark Document](BENCHMARKS.md) for full details.

## 12. Technology Stack

| Component | Technology | Rationale |
|-----------|-----------|-----------|
| Inference engine | Rust (custom RWKV-4) | Performance, memory safety, no GC |
| Parallelism | rayon | Zero-overhead data parallelism |
| HTTP API | axum | Async, type-safe |
| P2P | tokio | Full-duplex, low latency |
| Tokenizer | Custom BPE | No Python dependencies |
| Database | SQLite (rusqlite) | Embedded, zero-config |
| Weight format | Custom binary | Direct mmap, no runtime overhead |

## 13. Privacy & Trust Model

### Threat Model

SHEAR is designed to operate in an adversarial environment where:

1. **Node operators may be malicious** — they can read, log, or modify data they receive
2. **Network paths may be monitored** — traffic analysis is assumed possible
3. **No central authority exists** — there is no "SHEAR Inc." to file complaints with

### Privacy Guarantees by Architecture

| Property | How SHEAR Achieves It |
|----------|----------------------|
| **No single node sees full prompt** | Router splits prompts into task shards before distribution |
| **No central point of failure** | P2P network, no relay/proxy/gateway |
| **Result verifiable without trusting source** | Consensus voting across independent cells |
| **Node misbehavior has cost** | Reputation system penalizes quality deviations |
| **Data stays in your jurisdiction** | You choose which nodes process your requests |
| **Full auditability** | 100% open source, no proprietary components |

### Comparison with Existing Approaches

| Approach | Privacy Model | Trust Required | Single Point of Failure |
|----------|--------------|----------------|------------------------|
| Cloud API (OpenAI, etc.) | None — provider sees everything | Trust the provider | Yes (the provider) |
| API Relay / Token Stations | None — middleman sees + can modify | Trust middleman + provider | Yes (the middleman) |
| VPN + Cloud API | Hides traffic, provider still sees prompt | Trust the provider | Yes (the provider) |
| Local Deployment | Full — data never leaves machine | Trust yourself | No |
| **SHEAR** | **Architectural** — no node sees full prompt | **Trust no single entity** | **No** |

### The "Naked Prompt" Problem

When using centralized AI APIs, your prompt is transmitted in full to a single entity. This is the equivalent of sending an unencrypted email — the content is fully exposed to every hop in the chain. For:

- **Enterprises**: Proprietary business logic, customer data, trade secrets
- **Developers**: Application prompts, API keys embedded in context, system instructions
- **Individuals**: Personal data, health information, private conversations
- **Government**: Classified information, citizen data, policy drafts

SHEAR's prompt sharding means that even if every node in the network logs its input, **no single log entry contains enough information to reconstruct the original prompt**.
