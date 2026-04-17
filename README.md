<p align="center">
  <h1>🦅 OpenSHEAR</h1>
  <p><strong>Open · Decentralized · Self-Evolving · Parallel AI Reasoning Network</strong></p>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/language-Rust-orange" alt="Rust">
  <img src="https://img.shields.io/badge/license-Apache%202.0-blue" alt="License">
  <img src="https://img.shields.io/badge/status-early%20development-yellow" alt="Status">
</p>

---

## What is OpenSHEAR?

OpenSHEAR is an open-source, decentralized, self-evolving **parallel AI inference network**.

Unlike traditional distributed inference (which splits one model across GPUs), OpenSHEAR treats each network node as an independent reasoning unit. Requests are **sharded into parallel sub-tasks**, distributed across heterogeneous nodes, and merged back — achieving true parallelism that scales with the network.

**The core insight:** Current LLMs generate tokens serially (A → B → C → D). No matter how many nodes you have, latency is bounded by a single node's speed. OpenSHEAR breaks this chain — multiple nodes work on the same request simultaneously.

## Why OpenSHEAR?

| Problem | Current Approach | OpenSHEAR |
|---------|-----------------|-----------|
| Serial token generation | 1 node per request, bounded latency | Parallel sub-task generation, latency drops with more nodes |
| GPU monopoly | Only data centers can participate | Any device can contribute (CPU / GPU / hybrid) |
| Single-point failure | Centralized API servers | P2P network, no central authority |
| Static models | Fixed weights, manual updates | Self-evolving: nodes improve via federated LoRA fine-tuning |
| No incentive | Volunteer-only | Credits system + on-chain reputation |

## Architecture Overview

```
                        ┌─────────────────────────────────┐
                        │         Client Request           │
                        └─────────────┬───────────────────┘
                                      ▼
                        ┌─────────────────────────────────┐
                        │      Shard Planner (Router)      │
                        │  Analyze → Decompose → Dispatch  │
                        └──────┬──────┬──────┬──────┬─────┘
                               │      │      │      │
                    ┌──────────▼┐ ┌────▼────┐ ┌▼───────▼───────┐
                    │  Node L1  │ │ Node L2 │ │   Node L2     │
                    │  (CPU)    │ │ (GPU)   │ │   (GPU)       │
                    │  Draft    │ │ Expert  │ │   Expert      │
                    │  Tokens   │ │ A       │ │   B           │
                    └─────┬─────┘ └────┬────┘ └───────┬───────┘
                          │            │              │
                    ┌─────▼────────────▼──────────────▼───────┐
                    │        Result Merger / Aggregator       │
                    │    Vote · Rank · Combine · Verify       │
                    └─────────────────┬──────────────────────┘
                                      ▼
                        ┌─────────────────────────────────┐
                        │         Final Response            │
                        └─────────────────────────────────┘
```

## Core Concepts

### 1. Parallel Inference (Not Just Distributed)

Traditional distributed inference (vLLM, TGI) uses **tensor parallelism** or **pipeline parallelism** — splitting one model across GPUs. This improves throughput but NOT per-request latency.

OpenSHEAR uses **task-level parallelism**:

- **Draft parallelism**: Multiple nodes generate candidate token sequences simultaneously
- **Expert parallelism**: Different nodes specialize in different domains (code, math, language)
- **Chunk parallelism**: Long responses are split into segments, each computed independently
- **Vote consensus**: Multiple candidates are ranked and merged for higher quality

```
Latency comparison:
  Serial:     |████████████████████████████████████|  30s
  OpenSHEAR:  |████████████████|                    15s (2 nodes)
  OpenSHEAR:  |██████████|                            8s (4 nodes)
```

### 2. Five-Tier Node Hierarchy

Not all nodes are equal. OpenSHEAR assigns roles based on hardware capability:

| Tier | Name | Hardware | Role | Model Size |
|------|------|----------|------|------------|
| L0 | Collector | CPU only | Data collection, tokenization, embedding | 0.1-0.3B |
| L1 | Light Inference | CPU + 4GB RAM | Draft generation, simple tasks | 0.5-1.5B |
| L2 | Standard Inference | RTX 3060 (8GB) | Domain expert inference | 7B |
| L3 | Heavy Inference | RTX 3090/4090 (24GB) | Complex reasoning, CoT, code | 14B+ |
| L4 | Data Center | A100/H100 (80GB) | Backbone model, verification | 70B+ |

A single request flows through multiple tiers:

```
L0 (tokenize) → L1 (draft 3 candidates) → L2 (domain experts refine) → L4 (verify quality)
```

### 3. Heterogeneous Architecture (dMoE)

OpenSHEAR doesn't require all nodes to run the same model. Each node runs:

```
Node = Mini-Base (0.5B~3B) + Domain LoRA = One Expert
```

Supported architectures:
- **Transformer**: Qwen, LLaMA, Mistral (for high-quality generation)
- **RWKV**: O(1) inference, no KV cache, CPU-friendly (for scalable nodes)
- **Mamba**: Linear-time attention (for long-context tasks)

The Router selects the best architecture per sub-task.

### 4. Self-Evolution Loop

Nodes don't stay static. They improve over time:

```
  Observe → Reflect → Evolve → Verify
     │         │         │         │
     ▼         ▼         ▼         ▼
  Collect   Analyze   LoRA     Quality
  Results   Failures  Fine-tune Check
     │         │         │         │
     └─────────┴─────────┴─────────┘
              ↑                       │
              └──── Iterate ──────────┘
```

- Failed inferences are collected as training data
- Nodes fine-tune their LoRA adapters locally
- Improved LoRA snapshots are shared via P2P
- Quality is verified before adoption (requester annotation + redundancy consensus)

### 5. Credits & Reputation

Every inference earns or costs Credits:

- **Requester** pays Credits for inference
- **Worker node** earns Credits for contributing
- **Quality bonus**: Higher-rated responses earn more
- **Staking**: Nodes stake Credits to participate (slashing for poor quality)

Long-term: on-chain settlement via smart contracts for trustless coordination.

## Tech Stack

- **Core**: Rust (Router, P2P network, Credits settlement)
- **Inference**: Python + PyTorch (model loading, LoRA fine-tuning)
- **Communication**: TCP P2P with JSON framing, Gossip protocol for node discovery
- **Models**: RWKV (CPU-first), Qwen/LLaMA (GPU), Mamba (long-context)
- **Smart Contracts**: Solidity (Credits, Reputation, Governance)
- **API**: OpenAI-compatible (`/v1/chat/completions`)

## Project Structure

```
OpenSHEAR/
├── src-rs/                    # Rust core
│   └── decentral-ai-core/
│       ├── src/
│       │   ├── main.rs        # Node binary (HTTP + P2P)
│       │   ├── lib.rs         # Shared types
│       │   ├── router.rs      # Request routing + load balancing
│       │   └── network.rs     # P2P layer (TCP + Gossip)
│       └── Cargo.toml
├── scripts/
│   ├── rwkv_engine.py         # RWKV inference engine
│   ├── inference_worker.py    # Python worker (OpenAI-compatible API)
│   └── router_bridge.py       # Router ↔ Worker bridge
├── contracts/                 # Solidity smart contracts
│   └── DecentralAI.sol
├── data/                      # Training datasets
├── docs/                      # Design documents
│   ├── DESIGN.md
│   ├── ARCHITECTURE_RESEARCH.md
│   └── COMPETITIVE_ANALYSIS.md
└── README.md
```

## Current Status

- [x] Core architecture design
- [x] Rust Router + P2P network layer
- [x] Python RWKV-4 inference engine (CPU)
- [x] End-to-end test: Router → Worker → RWKV
- [x] HumanEval baseline (164 problems)
- [x] Smart contracts (Credits + Reputation + Settlement)
- [ ] Parallel inference engine (shard + dispatch + merge)
- [ ] Node self-evolution loop
- [ ] LoRA fine-tuning pipeline
- [ ] Credits settlement integration
- [ ] Multi-node parallel benchmark

## Quick Start (Coming Soon)

```bash
# Clone
git clone https://github.com/llianakindingucpu-stack/OpenSHEAR.git
cd OpenSHEAR

# Build Rust core
cd src-rs/decentral-ai-core
cargo build --release

# Start a node
./target/release/decentral-ai-core --role L1 --port 8080 --peers <peer_addr>
```

## Get Involved

OpenSHEAR is in early development. Contributions welcome:

- **Run a node**: Any device with CPU can be an L0/L1 node
- **Improve models**: Fine-tune LoRA adapters for your domain
- **Build tools**: Dashboard, monitoring, mobile client
- **Spread the word**: Star the repo, share the vision

## Vision

> Break AI monopoly. Build an AI network that cannot be shut down, cannot be controlled, and belongs to everyone.

---

## License

Apache License 2.0
