# Changelog

All notable changes to OpenSHEAR will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [0.2.0] - 2026-04-19

### Added
- **Speculative Decoding Engine** (`speculative.rs`): Multi-cell parallel draft + ensemble voting
  - Three voting strategies: Majority, ConfidenceWeighted, LeaderFollow
  - Rayon-parallelized cell execution (1.1x faster than single-cell baseline)
  - Configurable consensus threshold (min_consensus 0.0–1.0)
- **Ensemble HumanEval Benchmark** (`humaneval_ensemble_main.rs`): Baseline vs. ensemble comparison
- **HumanEval CLI** (`humaneval_main.rs`): Resume support, per-problem panic isolation

### Fixed
- **NaN overflow crash**: `decay.exp()` unbounded → added `.clamp(1e-30, 1e10)` in 3 locations
- **Weighted sum overflow**: Added intermediate value clamping `prod.clamp(-1e20, 1e20)`
- **Token ID out of bounds**: Added `safe_id()` bounds check before embedding lookup
- **full_code duplication bug**: Completion already includes prompt, no need to concatenate again

### Changed
- **Repo cleanup**: Removed 91 legacy files (Python, Solidity, docs), now pure Rust project (24 files)
- **README**: Complete rewrite with architecture diagram, benchmarks, build instructions
- **Documentation**: Added ARCHITECTURE.md, BENCHMARKS.md, SPECULATIVE_DECODING.md, NODE_HIERARCHY.md, CONTRIBUTING.md

### Benchmark Results
- HumanEval 164/164: Pass@1 = 0.0% (untrained 169M base, expected)
- Baseline speed: 4.87 tok/s (avg) on Intel Pentium G4560 CPU
- Ensemble (3 cells, rayon): 2.4 tok/s with 45.6% consensus rate
- Zero crashes across all benchmarks after NaN fix

## [0.1.0] - 2026-04-18

### Added
- **RWKV-4 Inference Engine** (`rwkv_model.rs`): Full WKV linear recurrence implementation
  - Custom binary weight format with direct loading
  - BPE tokenizer (`tokenizer.rs`) with full merge table
  - Top-p sampling with temperature control
- **SHEAR Cell** (`cell.rs`): Core inference unit with TimeMix + SwiGLU FFN
- **Aggregator** (`aggregator.rs`): WeightedSum, BestOfN, RankBased strategies
- **Router** (`router.rs`): Request routing with SQLite-based credit/reputation system
- **Network** (`network.rs`): P2P networking foundation with tokio
- **CLI** (`shear_main.rs`): Interactive inference + speculative decoding mode
- **Architecture Design Document** (`docs/SHEAR_DESIGN.md`)

### Technical Details
- RWKV-4-169M model: 12 layers, 768 hidden dim, 12 heads, 169.3M params
- Inference speed: ~3.0 tok/s CPU (debug), ~4.3-4.9 tok/s (release)
- Weight format: custom binary (header: vocab/n_layers/hidden/ffn/magic "RWKV")
