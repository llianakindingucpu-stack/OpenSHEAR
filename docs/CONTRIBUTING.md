# Contributing to OpenSHEAR

Thank you for your interest in contributing! This guide covers everything you need to get started.

## Quick Start

```bash
# Clone
git clone https://github.com/llianakindingucpu-stack/OpenSHEAR.git
cd OpenSHEAR/src-rs/decentral-ai-core

# Build
cargo build --release

# Test
cargo test

# Run
cargo run --release --bin shear -- --help
```

## Development Setup

### Prerequisites

- **Rust** 1.75+ (`rustup update stable`)
- **C compiler** (for rusqlite bundled mode — MSVC on Windows, gcc on Linux)
- **Git**

### Optional (for model testing)

- RWKV-4 model weights in binary format (see `rwkv_weights.rs` for format)
- BPE tokenizer data (`data/id2token.txt`, `data/merges.txt` — included in repo)

## Project Structure

```
src-rs/decentral-ai-core/
├── src/
│   ├── lib.rs              ← Library root (AppState, NodeRole, ChatMessage)
│   ├── cell.rs             ← SHEAR Cell inference unit
│   ├── aggregator.rs       ← Output aggregation (WeightedSum/BestOfN/RankBased)
│   ├── speculative.rs      ← Speculative decoding engine
│   ├── rwkv_model.rs       ← RWKV-4 inference (WKV recurrence)
│   ├── rwkv_weights.rs     ← Binary weight loader
│   ├── tokenizer.rs        ← BPE tokenizer
│   ├── network.rs          ← P2P networking
│   ├── router.rs           ← Request routing + Credits
│   ├── inference.rs        ← Inference abstraction
│   ├── shear_main.rs       ← Main CLI
│   ├── humaneval.rs        ← HumanEval benchmark core
│   ├── humaneval_main.rs   ← HumanEval CLI
│   └── humaneval_ensemble_main.rs ← Ensemble benchmark CLI
├── data/                   ← Tokenizer data
└── Cargo.toml
```

## Code Style

- **Rust standard**: `cargo fmt` before committing
- **No `unwrap()`** in production code — use `anyhow::Result` or explicit error handling
- **Comments**: Document public APIs with `///` doc comments. Complex algorithms need inline comments.
- **Testing**: Every new public function needs a unit test. Integration tests go in `tests/`.

## Commit Convention

```
type: short description

type = feat | fix | refactor | docs | test | perf | chore
```

Examples:
```
feat: add LeaderFollow voting strategy
fix: clamp decay.exp() to prevent NaN overflow
refactor: extract WKV recurrence into separate function
docs: update ARCHITECTURE.md with Aggregator details
test: add cell forward pass unit tests
perf: parallelize draft generation with rayon
```

## Pull Request Process

1. **Fork** the repository
2. **Create a feature branch**: `git checkout -b feat/my-feature`
3. **Make changes** with proper commit messages
4. **Test**: `cargo test` must pass
5. **Format**: `cargo fmt` must produce no changes
6. **Submit PR** with description of changes and motivation

## Areas That Need Help

### High Priority
- **GPU inference** (CUDA backend for RWKV-4)
- **Quantization** (INT4/INT8 for CPU-only nodes)
- **LoRA fine-tuning** integration
- **HumanEval evaluation** with trained models

### Medium Priority
- **Network protocol** design and implementation
- **Credits/Reputation system** smart contracts
- **Dashboard** for node monitoring
- **Python bindings** (PyO3)

### Low Priority / Nice to Have
- **WebAssembly** build for browser-based inference
- **Mobile** (Android/iOS) node support
- **Model conversion tools** (HF → SHEAR binary format)
- **Visualization** of ensemble voting patterns

## Communication

- **GitHub Issues**: Bug reports, feature requests
- **GitHub Discussions**: Design proposals, questions

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
