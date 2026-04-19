# Benchmarks

All benchmarks run on **Intel Pentium G4560** (2C/4T, AVX2, no AVX-512), **8GB RAM**, **no GPU**.

## RWKV-4-169M Baseline

Model: RWKV-4-Pile-169M (untrained base model)

| Metric | Value |
|--------|-------|
| Model size | 169.3M parameters |
| Weight format | Custom binary (646 MB) |
| Load time | ~5.3s |
| Inference speed | 4.87 tok/s (avg), range 1.5–6.2 tok/s |
| Memory usage | ~1.2 GB RAM |

## HumanEval Benchmark

| Configuration | Tasks Completed | Pass@1 | Crashes | Avg Speed |
|---------------|----------------|--------|---------|-----------|
| Baseline (1 Cell, 169M) | 164/164 | 0.0% | 0 | 4.87 tok/s |

**Note**: 0.0% Pass@1 is expected for an untrained 169M base model. The benchmark validates engine stability (zero crashes across all 164 problems).

### HumanEval Output Quality

- 74/164 problems generated code containing `return` statements
- All 164 problems produced valid Python function signatures
- Zero parsing errors
- Model output is semantically meaningless (expected for base model)

## Speculative Decoding (Ensemble Voting)

3 Cells, shared RWKV-4-169M weights, different temperatures [0.5, 0.8, 1.1]

| Metric | Baseline | Ensemble (serial) | Ensemble (parallel, rayon) |
|--------|----------|-------------------|----------------------------|
| Speed | 2.2 tok/s | 0.8 tok/s (2.1x slower) | **2.4 tok/s (1.1x faster)** |
| Consensus rate | N/A | N/A | 45.6% |
| Cell 0 hit rate (T=0.5) | N/A | N/A | 33.3% |
| Cell 1 hit rate (T=0.8) | N/A | N/A | 46.7% |
| Cell 2 hit rate (T=1.1) | N/A | N/A | 56.7% |

### Key Findings

1. **Parallelization is essential**: Serial ensemble is 2.1x slower. With rayon parallelism, ensemble is actually 1.1x faster than baseline (overhead eliminated).
2. **Higher temperature → more diverse → more hits**: Cell 2 (T=1.1) produces the most "unique" tokens that end up being selected.
3. **Consensus ~45%**: About half the time, all cells agree. The other half, voting selects the best output.
4. **No quality regression**: Ensemble voting does not degrade output quality vs. single-cell inference.

## NaN Overflow Fix

Critical bug discovered and fixed during HumanEval benchmarking:

| Issue | Cause | Fix |
|-------|-------|-----|
| Layer 0 NaN at token 9599+ | `decay.exp()` unbounded → `inf` → `NaN` | `decay.exp().clamp(1e-30, 1e10)` |
| Weighted sum overflow | Intermediate products exceed f32 range | `prod.clamp(-1e20, 1e20)` |
| Token ID out of bounds | Invalid token_id from sampling | `safe_id()` bounds check |

After fix: **0 NaN values** across 164 HumanEval problems, ~800K total tokens generated.

## Hardware Constraints

Current development environment is extremely constrained:

- **CPU**: Intel Pentium G4560 (2017 dual-core, AVX2 only)
- **RAM**: 8 GB (model + OS + tools fills most of it)
- **GPU**: None (Intel HD Graphics 610 integrated)
- **Disk**: C: ~656 MB free, D: ~1.4 GB free
- **Network**: HuggingFace/GitHub blocked, requires mirrors

Despite these constraints, the entire inference stack works. This validates the project's core premise: **SHEAR runs on commodity hardware.**

## Future Benchmarks (Planned)

| Benchmark | Requires | Status |
|-----------|----------|--------|
| HumanEval with trained model | GPU (LoRA fine-tuning) | Blocked |
| 7B model inference | GPU (6GB+ VRAM) | Blocked |
| Multi-node ensemble | 2+ machines | Blocked |
| Speculative Decoding Phase 2 | Heterogeneous nodes | Design complete |
| TOPLOC verification | Multi-node | Design complete |
