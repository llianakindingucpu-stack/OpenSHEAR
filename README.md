<div align="center">

# ⚡ OpenSHEAR

**Stateless Hybrid Ensemble Architecture for Reasoning**

[![Rust](https://img.shields.io/badge/Rust-1.75%2B-orange?logo=rust)](https://www.rust-lang.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](docs/CONTRIBUTING.md)

*Run AI inference on anything. From a $50 PC to a GPU cluster.*

[Architecture](docs/ARCHITECTURE.md) · [Benchmarks](docs/BENCHMARKS.md) · [Speculative Decoding](docs/SPECULATIVE_DECODING.md) · [Contributing](docs/CONTRIBUTING.md)

</div>

---

## Why OpenSHEAR?

Most AI inference frameworks need expensive, uniform hardware. OpenSHEAR is different:

- 🧩 **Heterogeneous** — Mix CPUs, gaming GPUs, and datacenter hardware in the same network
- ⚡ **Speculative Decoding** — Fast nodes draft, accurate nodes verify, 1.5-2× speedup
- 🔌 **No KV Cache** — Each Cell uses linear recurrence (O(1)/token), not attention
- 🛡️ **Fault Tolerant** — Remove any Cell; the rest keep working
- 🦀 **Pure Rust** — Zero Python dependency, no GC pauses, memory-safe

## How It Works

```
                    User Request
                         │
         ┌───────────────▼───────────────┐
         │         Aggregator             │
         │  WeightedSum · BestOfN · Rank  │
         └──┬──────────┬──────────┬──────┘
            │          │          │
       ┌────▼──┐  ┌───▼───┐  ┌──▼────┐
       │Cell 0 │  │Cell 1 │  │Cell N │   ← All parallel
       │ 200M  │  │ 200M  │  │ 200M  │     No cross-talk
       │local  │  │local  │  │local  │     Local state only
       │state  │  │state  │  │state  │
       └───────┘  └───────┘  └───────┘
```

Each **Cell** runs an independent RWKV-4 model. The **Aggregator** merges outputs through competitive selection — not averaging, but picking the best. This is how the cerebral cortex works: billions of cortical columns, each independent, coordinated sparsely.

## Quick Start

```bash
# Build
git clone https://github.com/llianakindingucpu-stack/OpenSHEAR.git
cd OpenSHEAR/src-rs/decentral-ai-core
cargo build --release

# Interactive inference
cargo run --release --bin shear -- --model /path/to/model.bin --prompt "Hello world"

# Speculative decoding (3 cells)
cargo run --release --bin shear -- --speculative --cells 3 --model /path/to/model.bin
```

## Benchmarks

Tested on **Intel Pentium G4560** (2-core CPU from 2017), 8GB RAM, **no GPU**:

| Metric | Single Cell | Ensemble (3 Cells) |
|--------|-------------|---------------------|
| Speed | 4.87 tok/s | 2.4 tok/s (parallel) |
| Consensus | — | 45.6% |
| HumanEval 164/164 | ✅ Zero crashes | ✅ Zero crashes |

→ [Full benchmark details](docs/BENCHMARKS.md)

## Node Hierarchy

Any hardware can participate:

| Level | Hardware | Model | Role |
|-------|----------|-------|------|
| **L0** | CPU only | None | Data collection, routing |
| **L1** | CPU + 4GB | 0.5-1.5B | Draft generation |
| **L2** | GPU 8GB | 7B | Standard inference |
| **L3** | GPU 24GB | 14B+ | Heavy inference, fine-tuning |
| **L4** | A100/H100 | 70B+ | Training, research |

→ [Node hierarchy details](docs/NODE_HIERARCHY.md)

## Project Structure

```
src-rs/decentral-ai-core/
├── Cargo.toml
├── src/
│   ├── cell.rs             ← SHEAR Cell (RWKV inference unit)
│   ├── aggregator.rs       ← Output aggregation strategies
│   ├── speculative.rs      ← Speculative decoding engine
│   ├── rwkv_model.rs       ← RWKV-4 inference (WKV recurrence)
│   ├── rwkv_weights.rs     ← Binary weight loader
│   ├── tokenizer.rs        ← BPE tokenizer
│   ├── network.rs          ← P2P networking
│   ├── router.rs           ← Request routing + Credits
│   ├── shear_main.rs       ← CLI entry point
│   └── ...
└── data/
    ├── id2token.txt        ← Token vocabulary
    └── merges.txt          ← BPE merge table
```

## Documentation

| Document | Description |
|----------|-------------|
| [Architecture](docs/ARCHITECTURE.md) | Detailed architecture with diagrams |
| [SHEAR Design](docs/SHEAR_DESIGN.md) | Core design principles and rationale |
| [Speculative Decoding](docs/SPECULATIVE_DECODING.md) | Draft-verify mechanism design |
| [Node Hierarchy](docs/NODE_HIERARCHY.md) | Five-level node system |
| [Benchmarks](docs/BENCHMARKS.md) | Performance results and analysis |
| [Contributing](docs/CONTRIBUTING.md) | How to contribute |
| [Changelog](CHANGELOG.md) | Version history |

## Roadmap

- [x] RWKV-4 inference engine (Rust)
- [x] Cell + Aggregator architecture
- [x] Speculative decoding (Phase 1: ensemble voting)
- [x] HumanEval benchmark (164/164, zero crashes)
- [ ] GPU inference backend (CUDA)
- [ ] LoRA fine-tuning support
- [ ] Speculative decoding Phase 2 (cross-node)
- [ ] P2P network protocol
- [ ] Credits & Reputation smart contracts
- [ ] 7B+ model support

## License

MIT
