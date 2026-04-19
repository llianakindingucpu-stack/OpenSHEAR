# OpenSHEAR

**Stateless Hybrid Ensemble Architecture for Reasoning**

Distributed AI inference network with speculative decoding, ensemble voting, and heterogeneous node support. Built in Rust for performance.

## Architecture

```
┌─────────────────────────────────────────┐
│              Aggregator                  │
│   (WeightedSum / BestOfN / RankBased)    │
└────────┬────────┬────────┬──────────────┘
         │        │        │
    ┌────▼───┐ ┌──▼────┐ ┌▼───────┐
    │ Cell 0 │ │ Cell 1 │ │ Cell N │
    │(RWKV-4)│ │(RWKV-4)│ │(...)   │
    └────────┘ └────────┘ └────────┘
```

Each **Cell** runs an RWKV-4 model independently. The **Aggregator** combines outputs via voting/weighting. **Speculative Decoding** enables parallel draft generation across cells.

## Features

- **RWKV-4 Inference Engine** — Custom Rust implementation with WKV linear recurrence (O(1) per token)
- **Speculative Decoding** — Multi-cell parallel draft + verification (rayon-accelerated)
- **Ensemble Voting** — Majority, weighted sum, and rank-based aggregation strategies
- **HumanEval Benchmark** — Integrated benchmark runner with resume support
- **BPE Tokenizer** — Full BPE tokenizer with merge table support

## Benchmark Results

| Metric | Baseline (1 Cell) | Ensemble (3 Cells) |
|--------|-------------------|---------------------|
| Speed (CPU, 169M) | ~4.9 tok/s | ~4.4 tok/s (1.1x parallel) |
| Consensus Rate | N/A | 45.6% |
| HumanEval Pass@1 | 0.0% (untrained) | TBD |

*Tested on Intel Pentium G4560, 8GB RAM, no GPU. 169M base model.*

## Build

```bash
cd src-rs/decentral-ai-core
cargo build --release
```

## Usage

```bash
# Interactive inference
cargo run --release --bin shear -- --model /path/to/rwkv.bin --prompt "Hello"

# Speculative decoding with 3 cells
cargo run --release --bin shear -- --speculative --model /path/to/rwkv.bin --cells 3

# HumanEval benchmark
cargo run --release --bin humaneval -- --model /path/to/rwkv.bin --data /path/to/HumanEval.jsonl

# Ensemble benchmark (baseline + ensemble comparison)
cargo run --release --bin humaneval_ensemble -- --model /path/to/rwkv.bin
```

## Project Structure

```
src-rs/decentral-ai-core/
├── Cargo.toml
├── src/
│   ├── lib.rs              # Library root
│   ├── cell.rs             # SHEAR Cell (RWKV inference unit)
│   ├── aggregator.rs       # Output aggregation strategies
│   ├── speculative.rs      # Speculative decoding engine
│   ├── rwkv_model.rs       # RWKV-4 inference (WKV recurrence)
│   ├── rwkv_weights.rs     # Binary weight loader
│   ├── tokenizer.rs         # BPE tokenizer
│   ├── network.rs           # P2P networking (planned)
│   ├── router.rs            # Request routing + credits
│   ├── inference.rs         # Inference abstraction layer
│   ├── shear_main.rs        # CLI entry point
│   ├── humaneval.rs         # HumanEval benchmark core
│   ├── humaneval_main.rs    # HumanEval CLI
│   ├── humaneval_ensemble_main.rs  # Ensemble benchmark
│   └── ...
└── data/
    ├── id2token.txt         # Token vocabulary
    └── merges.txt           # BPE merge table
```

## Design Docs

- [SHEAR Architecture Design](docs/SHEAR_DESIGN.md)

## License

MIT
