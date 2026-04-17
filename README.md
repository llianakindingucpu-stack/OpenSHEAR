<p align="center">
  <h1>🦅 SHEAR</h1>
  <p><strong>Stateless Hybrid Ensemble Architecture for Reasoning</strong></p>
  <p>无状态 · 混合集成 · 并行大模型推理引擎</p>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/language-Rust-orange" alt="Rust">
  <img src="https://img.shields.io/badge/architecture-Ensemble-green" alt="Ensemble">
  <img src="https://img.shields.io/badge/license-Apache%202.0-blue" alt="License">
  <img src="https://img.shields.io/badge/status-design%20phase-yellow" alt="Status">
</p>

---

## What is SHEAR?

SHEAR is a **parallel inference engine** that fundamentally rethinks how large language models work.

Current LLMs generate tokens **serially** — layer 1 → layer 2 → ... → layer N → next token. No matter how many GPUs you add, latency is bounded by a single chain. This is why only data centers with hundreds of GB of VRAM can play.

**SHEAR breaks the chain.**

Instead of one deep model, SHEAR uses **many small, independent Cells running in parallel**. Each Cell processes the same input simultaneously, and an Aggregator merges their outputs into the final token. No layer dependencies. No KV cache. No sequential bottleneck.

```
Traditional LLM:          SHEAR:

Token → [L1→L2→...→L12] → Token    Token → [Cell A][Cell B]...[Cell N] → Aggregator → Token
         ╰── serial, 1 node ──╯                    ╰──── parallel, N nodes ────╯
```

**Result:** Latency drops as you add more nodes. A 100-Mbit network is enough. A 2GB machine can run a Cell. Anyone can participate.

---

## The Problem We Solve

| Problem | Status Quo | SHEAR |
|---------|-----------|-------|
| **Latency** | Bounded by sequential depth | Drops with more nodes |
| **Hardware** | Requires multi-GPU data centers | Runs on commodity hardware |
| **Scalability** | Vertical (bigger GPU) | Horizontal (more nodes) |
| **Availability** | Single provider, can be shut down | Decentralized, no single point of failure |
| **Training cost** | Millions of dollars | Thousands (per Cell) |
| **GPU monopoly** | Only rich orgs can train | Anyone can train a Cell |

---

## Architecture

### Core Design Principles (Iron Laws)

1. **No layer dependencies** — No L1→L2→L3 serial chain
2. **No global state** — No KV cache, no cross-Cell synchronization
3. **Fully sharded** — Any subset of Cells can produce a valid output
4. **Preemption-safe** — Redundant dispatch, first result wins
5. **Minimal communication** — Per-shard result ≤ 512 bytes

### The Model

```
                          ┌──────────────────┐
                          │  Token Embedding  │
                          └────────┬─────────┘
                                   │ Broadcast
            ┌──────────┬───────────┼───────────┬──────────┐
            ▼          ▼           ▼           ▼          ▼
        ┌───────┐  ┌───────┐  ┌───────┐  ┌───────┐  ┌───────┐
        │Cell A │  │Cell B │  │Cell C │  │  ...  │  │Cell N │
        │~200M  │  │~200M  │  │~200M  │  │       │  │~200M  │
        │ Code  │  │ Math  │  │ Lang  │  │       │  │ Logic │
        └───┬───┘  └───┬───┘  └───┬───┘  └───┬───┘  └───┬───┘
            │          │           │           │          │
            └──────────┴───────────┼───────────┴──────────┘
                                   ▼
                          ┌──────────────────┐
                          │   Aggregator      │
                          │ Weighted Fusion   │
                          │ (Learned Routing) │
                          └────────┬─────────┘
                                   ▼
                          ┌──────────────────┐
                          │  Next Token Prob  │
                          └──────────────────┘
```

**Every Cell is:**
- An independent feedforward neural network (~200M params)
- With local linear recurrence (RWKV-style time-mix) for sequence understanding
- Optionally specialized: code, math, language, logic, etc.
- Runnable on a single machine with 2GB RAM
- Stateless to other Cells — no cross-Cell communication needed

**32 Cells × 200M = 6.4B total params.** Each node only needs 200M. The network provides the rest.

### Why This Works

**Ensemble theory** predicts that diverse weak learners combined can match or exceed a single strong learner. This is not speculation — it's math:

- DeepSeek V3 (256 Experts, 671B total): GPT-4 level performance
- GPT-4 itself uses Mixture of Experts internally
- Brain cortical columns: billions of weak processors = general intelligence

**SHEAR's key insight:** MoE works because experts are parallel. But current MoE runs all experts on ONE machine. SHEAR distributes them across a NETWORK.

| | MoE (DeepSeek V3) | SHEAR |
|---|---|---|
| Expert location | Same GPU cluster | Any machine, anywhere |
| Communication | NVLink, 100Gbps | Internet, 100Mbps |
| Hardware barrier | 8×H100 (~$200K) | 32× laptop (~$32K) |
| Who can run | Google, DeepSeek | You |

### Speculative Parallelism

For each token, ALL Cells generate candidates simultaneously. The Aggregator picks the best one:

```
Time →
Cell A: ████████████░░░░  (finished at t=12, confidence=0.7)
Cell B: ██████████████░░  (finished at t=14, confidence=0.85) ← Best, selected
Cell C: ██████░░░░░░░░░░  (crashed, ignored — preempt-safe)
...
```

First valid result wins. No waiting. No bottleneck.

---

## Tech Stack

| Layer | Technology | Why |
|-------|-----------|-----|
| **Core framework** | Rust | Performance, safety, single binary deployment |
| **Cell inference** | Rust (candle) | No Python dependency at runtime |
| **Cell training** | Python + PyTorch | Mature ecosystem, rapid prototyping |
| **Communication** | TCP P2P + Gossip | No central coordinator, self-organizing |
| **Smart contracts** | Solidity | On-chain Credits + Reputation settlement |
| **API** | OpenAI-compatible | Drop-in replacement for existing tools |

## Project Structure

```
SHEAR/
├── src-rs/
│   └── shear-core/
│       ├── src/
│       │   ├── main.rs          # Cell server binary
│       │   ├── lib.rs           # Shared types
│       │   ├── cell.rs          # Cell inference engine
│       │   ├── aggregator.rs    # Result aggregation
│       │   ├── router.rs        # Task routing + load balancing
│       │   └── network.rs       # P2P layer (TCP + Gossip)
│       └── Cargo.toml
├── train/                        # Cell training pipeline
│   ├── train_cell.py            # Single Cell trainer
│   ├── train_aggregator.py      # Aggregator weight trainer
│   └── dataset/                 # Training data
├── contracts/                    # Solidity smart contracts
│   └── SHEAR.sol               # Credits + Reputation + Settlement
├── docs/                         # Design documents
│   ├── SHEAR_DESIGN.md          # Architecture spec
│   ├── DESIGN.md                # Original DecentralAI design
│   └── COMPETITIVE_ANALYSIS.md  # Competitive landscape
└── README.md
```

## Roadmap

### Phase 0 — Foundation (Current)
- [x] Architecture design and mathematical formulation
- [x] P2P network layer (TCP + Gossip)
- [x] Router + load balancing (Rust)
- [x] Python RWKV-4 inference (proof of concept)
- [ ] **Cell inference engine (Rust)**
- [ ] **Aggregator implementation**

### Phase 1 — Working Prototype
- [ ] Train first Cell (small scale, toy dataset)
- [ ] Multi-Cell parallel inference (2-4 Cells, same machine)
- [ ] Aggregator with learned routing weights
- [ ] End-to-end benchmark: Cells → Aggregator → Output

### Phase 2 — Distributed
- [ ] Multi-node parallel inference across network
- [ ] Preemption and fault tolerance
- [ ] Cell specialization (code / math / language)
- [ ] Credits system integration

### Phase 3 — Self-Evolution
- [ ] LoRA fine-tuning pipeline per Cell
- [ ] Cell snapshot sharing via P2P
- [ ] Quality verification loop
- [ ] Reputation + incentive system

### Phase 4 — Production
- [ ] On-chain Credits settlement (Solidity)
- [ ] Public testnet with 10+ nodes
- [ ] OpenAI-compatible API gateway
- [ ] Node onboarding tool (one-click deploy)

## Quick Start (Coming Soon)

```bash
git clone https://github.com/llianakindingucpu-stack/OpenSHEAR.git
cd OpenSHEAR

# Build
cd src-rs/shear-core && cargo build --release

# Run a Cell
./target/release/shear-core --cell-id A --port 9001

# Run the Aggregator
./target/release/shear-core --role aggregator --cells localhost:9001,localhost:9002
```

## Contributing

SHEAR is early-stage. The hardest problems are ahead of us. We're looking for:

- **ML Engineers** — Cell architecture, training pipeline, evaluation
- **Systems Engineers** — Distributed computing, P2P networking
- **Mathematicians** — Ensemble theory, information fusion, convergence proofs
- **Visionaries** — Anyone who believes AI should belong to everyone

See [CONTRIBUTING.md](./CONTRIBUTING.md) for guidelines.

## Vision

> AI shouldn't require a billion dollars and a data center.  
> It should require a laptop, an internet connection, and the will to contribute.  
>  
> SHEAR makes every node an expert. The network is the model.

---

## License

Apache License 2.0
